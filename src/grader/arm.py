"""Arm controller for the robot fruit sorting system.

Controls an Arctos robot arm (3-axis XYZ only) via CAN bus with MKS SERVO42D
stepper motors.  Joints 4-6 are removed and replaced with a fixed 8-inch
(~203 mm) suction tube.  Provides waypoint navigation, pressure-based descent,
and workspace-bounds validation.
"""

from __future__ import annotations

import logging
import math
import struct
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import can

logger = logging.getLogger(__name__)


class ArmController:
    """High-level controller for the Arctos 3-axis (XYZ) robot arm.

    Joints 4-6 are removed; a fixed 8-inch (~203 mm) suction tube is mounted
    in their place.  The tube length is accounted for as a Z offset so that
    waypoint Z values represent the height of the suction cup tip, not the
    arm flange.
    """

    # Length of the suction tube mounted in place of joints 4-6 (mm).
    SUCTION_TUBE_LENGTH: float = 203.0  # 8 inches

    def __init__(self, config: dict, mock_mode: bool = False) -> None:
        self._mock_mode = mock_mode

        self._can_interface: str = config["can_interface"]
        self._can_bitrate: int = config["can_bitrate"]
        self._motor_ids: list[int] = list(config["motor_ids"])
        self._max_velocity: float = float(config["max_velocity"])
        self._max_acceleration: float = float(config["max_acceleration"])
        self._safe_z_height: float = float(config["safe_z_height"])
        self._descent_speed: float = float(config["descent_speed"])
        self._tube_length: float = float(
            config.get("suction_tube_length", self.SUCTION_TUBE_LENGTH)
        )

        bounds = config["workspace_bounds"]
        self._bounds_x: tuple[float, float] = tuple(bounds["x"])
        self._bounds_y: tuple[float, float] = tuple(bounds["y"])
        self._bounds_z: tuple[float, float] = tuple(bounds["z"])

        self._waypoints: dict[str, tuple[float, float, float]] = {
            name: tuple(wp) for name, wp in config["waypoints"].items()
        }

        pz = config["pick_zone"]
        self._pick_zone_x: tuple[float, float] = tuple(pz["x_range"])
        self._pick_zone_y: tuple[float, float] = tuple(pz["y_range"])

        # Start at home position
        home = self._waypoints["home"]
        self._position: tuple[float, float, float] = home

        self._bus: can.BusABC | None = None

    # -- Context manager -------------------------------------------------------

    def __enter__(self) -> ArmController:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.disconnect()

    # -- Lifecycle -------------------------------------------------------------

    def connect(self) -> None:
        """Open the CAN bus connection (or log in mock mode)."""
        if self._mock_mode:
            logger.info("ArmController connected in mock mode — no real CAN bus")
            return

        import can

        self._bus = can.interface.Bus(
            channel=self._can_interface,
            interface="socketcan",
            bitrate=self._can_bitrate,
        )
        logger.info(
            "CAN bus opened on %s at %d bps", self._can_interface, self._can_bitrate
        )

    def disconnect(self) -> None:
        """Shut down the CAN bus connection."""
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None
            logger.info("CAN bus shut down")
        else:
            logger.info("ArmController disconnected (mock or already closed)")

    # -- Motion ----------------------------------------------------------------

    def move_to(
        self, x: float, y: float, z: float, speed: float | None = None
    ) -> None:
        """Move the arm to the target XYZ position.

        Parameters
        ----------
        x, y, z:
            Target position in workspace coordinates (mm).
        speed:
            Override speed (mm/s).  Defaults to *max_velocity*.

        Raises
        ------
        ValueError
            If the target is outside workspace bounds.
        """
        self._validate_bounds(x, y, z)
        speed = speed if speed is not None else self._max_velocity

        if self._mock_mode:
            travel_time = self._compute_travel_time(x, y, z, speed)
            logger.info(
                "Mock move to (%.2f, %.2f, %.2f) at %.1f mm/s — "
                "simulated %.3f s travel",
                x, y, z, speed, travel_time,
            )
            time.sleep(min(travel_time, 0.01))  # Cap mock sleep to keep tests fast
            self._position = (x, y, z)
            return

        joint_angles = self._ik_solve(x, y, z)
        self._send_joint_positions(joint_angles)
        self._position = (x, y, z)
        logger.info("Moved to (%.2f, %.2f, %.2f)", x, y, z)

    def get_position(self) -> tuple[float, float, float]:
        """Return the current XYZ position."""
        return self._position

    def home(self) -> None:
        """Move to the stored HOME waypoint."""
        self.move_to_waypoint("home")

    def emergency_stop(self) -> None:
        """Immediately stop all motors."""
        logger.warning("EMERGENCY STOP triggered")
        if self._mock_mode:
            logger.info("Mock emergency stop — all motors halted")
            return

        import can

        for motor_id in self._motor_ids:
            stop_msg = can.Message(
                arbitration_id=motor_id,
                data=b"\x00\x00\x00\x00\x00\x00\x00\x00",
                is_extended_id=False,
            )
            if self._bus is not None:
                self._bus.send(stop_msg)
        logger.info("Emergency stop commands sent to all %d motors", len(self._motor_ids))

    def move_to_waypoint(self, name: str) -> None:
        """Move to a named waypoint.

        Parameters
        ----------
        name:
            One of the configured waypoint names (e.g. ``home``, ``inspect``,
            ``chute_a``, ``chute_b``, ``chute_c``).

        Raises
        ------
        KeyError
            If the waypoint name is not found.
        """
        if name not in self._waypoints:
            raise KeyError(
                f"Unknown waypoint '{name}'. "
                f"Available: {', '.join(sorted(self._waypoints))}"
            )
        x, y, z = self._waypoints[name]
        logger.info("Moving to waypoint '%s' at (%.2f, %.2f, %.2f)", name, x, y, z)
        self.move_to(x, y, z)

    def estimate_travel_time(
        self, target_x: float, target_y: float, target_z: float
    ) -> float:
        """Estimate travel time in seconds from current position to target.

        Uses simple distance / max_velocity calculation.
        """
        return self._compute_travel_time(
            target_x, target_y, target_z, self._max_velocity
        )

    def is_in_pick_zone(self, x: float, y: float) -> bool:
        """Check whether the given XY coordinate is within the pick zone."""
        return (
            self._pick_zone_x[0] <= x <= self._pick_zone_x[1]
            and self._pick_zone_y[0] <= y <= self._pick_zone_y[1]
        )

    def descend_to_contact(
        self,
        target_x: float,
        target_y: float,
        start_z: float,
        pressure_callback: Callable[[], bool],
    ) -> float:
        """Descend from *start_z* until the pressure callback signals contact.

        Parameters
        ----------
        target_x, target_y:
            XY position to hold during descent.
        start_z:
            Starting Z height (mm).
        pressure_callback:
            Callable returning ``True`` when contact is detected.

        Returns
        -------
        float
            The Z position where contact was detected.
        """
        step_mm = self._descent_speed * 0.01  # 10 ms steps
        current_z = start_z

        if self._mock_mode:
            mock_contact_steps = 5
            for step in range(mock_contact_steps):
                current_z -= step_mm
                logger.debug("Mock descent step %d — z=%.2f", step, current_z)
                time.sleep(0.01)
            logger.info(
                "Mock contact detected at z=%.2f (after %d steps)",
                current_z,
                mock_contact_steps,
            )
            self._position = (target_x, target_y, current_z)
            return current_z

        self._validate_bounds(target_x, target_y, current_z)
        while current_z > self._bounds_z[0]:
            current_z -= step_mm
            self.move_to(target_x, target_y, current_z, speed=self._descent_speed)
            if pressure_callback():
                logger.info("Contact detected at z=%.2f", current_z)
                self._position = (target_x, target_y, current_z)
                return current_z

        logger.warning("Reached minimum Z (%.2f) without contact", self._bounds_z[0])
        self._position = (target_x, target_y, current_z)
        return current_z

    # -- Inverse kinematics ----------------------------------------------------

    def _ik_solve(self, x: float, y: float, z: float) -> list[float]:
        """Compute joint positions for a target XYZ position.

        The arm uses only 3 axes (X, Y, Z).  The suction tube hangs straight
        down from the Z carriage, so the arm's Z motor must target
        ``z + tube_length`` to place the cup tip at the requested *z*.

        .. todo::
            Replace this placeholder with real IK using the Arctos DH
            parameters once the arm hardware is available.

        Returns
        -------
        list[float]
            Three motor positions ``[x, y, z_motor]``.
        """
        # TODO: Implement real IK with Arctos DH parameters when hardware is available.
        z_motor = z + self._tube_length
        return [x, y, z_motor]

    def _send_joint_positions(self, angles: list[float]) -> None:
        """Send position commands to each motor over CAN.

        Parameters
        ----------
        angles:
            List of joint angles (one per motor).
        """
        if self._mock_mode:
            logger.debug("Mock CAN send — joint angles: %s", angles)
            return

        import can

        for motor_id, angle in zip(self._motor_ids, angles):
            position_bytes = struct.pack(">f", angle)
            data = position_bytes + b"\x00" * (8 - len(position_bytes))
            msg = can.Message(
                arbitration_id=motor_id,
                data=data,
                is_extended_id=False,
            )
            if self._bus is not None:
                self._bus.send(msg)
        logger.debug("Joint position commands sent to %d motors", len(self._motor_ids))

    # -- Internal helpers ------------------------------------------------------

    def _validate_bounds(self, x: float, y: float, z: float) -> None:
        """Raise ``ValueError`` if the position is outside workspace bounds."""
        if not (self._bounds_x[0] <= x <= self._bounds_x[1]):
            raise ValueError(
                f"X={x:.2f} is out of bounds [{self._bounds_x[0]}, {self._bounds_x[1]}]"
            )
        if not (self._bounds_y[0] <= y <= self._bounds_y[1]):
            raise ValueError(
                f"Y={y:.2f} is out of bounds [{self._bounds_y[0]}, {self._bounds_y[1]}]"
            )
        if not (self._bounds_z[0] <= z <= self._bounds_z[1]):
            raise ValueError(
                f"Z={z:.2f} is out of bounds [{self._bounds_z[0]}, {self._bounds_z[1]}]"
            )

    def _compute_travel_time(
        self, x: float, y: float, z: float, speed: float
    ) -> float:
        """Compute travel time from current position to (x, y, z)."""
        cx, cy, cz = self._position
        distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
        if speed <= 0:
            return 0.0
        return distance / speed
