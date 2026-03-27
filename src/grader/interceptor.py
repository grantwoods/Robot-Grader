"""Interception calculator — predicts where to pick fruit on a moving belt."""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class InterceptionCalculator:
    """Calculates the XYZ position where the arm should intercept a moving fruit."""

    def __init__(self, safe_z_height: float = 150.0, max_iterations: int = 3) -> None:
        self._safe_z = safe_z_height
        self._max_iter = max_iterations

    def calculate_intercept(
        self,
        fruit_pos: tuple[float, float],
        fruit_velocity: tuple[float, float],
        arm_travel_time_fn: Callable[[float, float, float], float],
        pick_zone_x_range: tuple[float, float],
        pick_zone_y_range: tuple[float, float],
    ) -> Optional[tuple[float, float, float]]:
        """Predict where the arm should go to intercept a fruit on the belt.

        Returns (x, y, z) target or None if the fruit will exit the pick zone
        before the arm can reach it.
        """
        fx, fy = fruit_pos
        vx, vy = fruit_velocity
        speed = math.sqrt(vx * vx + vy * vy)
        if speed < 0.1:
            return (fx, fy, self._safe_z)

        # Iterative refinement: arm travel time depends on target position,
        # which depends on how long the fruit travels, which depends on arm
        # travel time. Converge in a few iterations.
        target_x, target_y = fx, fy
        for _ in range(self._max_iter):
            travel_time = arm_travel_time_fn(target_x, target_y, self._safe_z)
            target_x = fx + vx * travel_time
            target_y = fy + vy * travel_time

        if not self._in_zone(target_x, target_y, pick_zone_x_range, pick_zone_y_range):
            logger.debug(
                "Intercept (%.1f, %.1f) is outside pick zone — skipping fruit",
                target_x, target_y,
            )
            return None

        return (target_x, target_y, self._safe_z)

    @staticmethod
    def _in_zone(
        x: float,
        y: float,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
    ) -> bool:
        return x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]
