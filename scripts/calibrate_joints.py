#!/usr/bin/env python3
"""Joint range-of-motion calibration tool.

Moves each axis (X, Y, Z) individually from its configured minimum bound to
its maximum bound and back, pausing at each end so the operator can verify
limits, direction, and clearance.

Usage:
    python scripts/calibrate_joints.py --mock
    python scripts/calibrate_joints.py --speed 30
    python scripts/calibrate_joints.py --axis z          # single axis only
"""

from __future__ import annotations

import argparse
import time

from grader.arm import ArmController
from grader.config import load_config

AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep each arm axis across its full range for calibration.",
    )
    parser.add_argument(
        "--config-dir", default="config",
        help="Config directory (default: config).",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock arm (no real hardware).",
    )
    parser.add_argument(
        "--speed", type=float, default=None,
        help="Override travel speed in mm/s (default: use config max_velocity).",
    )
    parser.add_argument(
        "--axis", choices=AXES, default=None,
        help="Calibrate a single axis only (default: all three).",
    )
    parser.add_argument(
        "--pause", type=float, default=2.0,
        help="Seconds to pause at each end of travel (default: 2).",
    )
    return parser.parse_args()


def sweep_axis(
    arm: ArmController,
    axis: str,
    bounds: tuple[float, float],
    speed: float | None,
    pause: float,
) -> None:
    """Move one axis from min to max and back, holding the others at home."""
    home_x, home_y, home_z = arm.get_position()
    lo, hi = bounds

    def pos(val: float) -> tuple[float, float, float]:
        if axis == "x":
            return (val, home_y, home_z)
        elif axis == "y":
            return (home_x, val, home_z)
        else:
            return (home_x, home_y, val)

    print(f"\n--- {axis.upper()} axis: {lo:.1f} mm -> {hi:.1f} mm ---")

    # Move to minimum
    print(f"  Moving to {axis.upper()} min ({lo:.1f})...")
    arm.move_to(*pos(lo), speed=speed)
    print(f"  At {axis.upper()} min. Pausing {pause:.0f}s — check clearance.")
    time.sleep(pause)

    # Move to maximum
    print(f"  Moving to {axis.upper()} max ({hi:.1f})...")
    arm.move_to(*pos(hi), speed=speed)
    print(f"  At {axis.upper()} max. Pausing {pause:.0f}s — check clearance.")
    time.sleep(pause)

    # Return to home on this axis
    print(f"  Returning {axis.upper()} to home...")
    arm.move_to(home_x, home_y, home_z, speed=speed)


def main() -> None:
    args = parse_args()
    cfg = load_config(config_dir=args.config_dir)
    arm_cfg = cfg["arm"]
    arm = ArmController(arm_cfg, mock_mode=args.mock)

    bounds_map = {
        "x": tuple(arm_cfg["workspace_bounds"]["x"]),
        "y": tuple(arm_cfg["workspace_bounds"]["y"]),
        "z": tuple(arm_cfg["workspace_bounds"]["z"]),
    }

    axes_to_run = [args.axis] if args.axis else list(AXES)

    print("=== Joint Range-of-Motion Calibration ===")
    print(f"Mock mode: {args.mock}")
    print(f"Speed: {args.speed or arm_cfg['max_velocity']} mm/s")
    print(f"Pause at limits: {args.pause}s")
    print(f"Axes: {', '.join(a.upper() for a in axes_to_run)}")
    for a in axes_to_run:
        lo, hi = bounds_map[a]
        print(f"  {a.upper()}: [{lo:.1f}, {hi:.1f}] mm")

    arm.connect()
    try:
        print("\nHoming first...")
        arm.home()
        print(f"  Home position: {arm.get_position()}")

        input("\nPress Enter to begin calibration sweeps (Ctrl+C to abort)...")

        for axis in axes_to_run:
            sweep_axis(arm, axis, bounds_map[axis], args.speed, args.pause)

        print("\nCalibration sweeps complete. Arm is at home position.")
        print("\nIf any axis moved in the wrong direction or hit a physical")
        print("limit before reaching the configured bound, adjust the values")
        print("in config/default.yaml under arm.workspace_bounds.")

    except KeyboardInterrupt:
        print("\n\nAborted by operator. Homing arm...")
        arm.home()
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
