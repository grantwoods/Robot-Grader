#!/usr/bin/env python3
"""Interactive camera-to-arm calibration tool.

Computes a homography mapping pixel coordinates to arm workspace coordinates
by collecting corresponding point pairs from the user.

Usage:
    python scripts/calibrate_camera.py
    python scripts/calibrate_camera.py --mock --output config/calibration_result.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from grader.camera import CameraManager
from grader.config import load_config

# Mutable state shared with the mouse callback.
_clicked_point: tuple[int, int] | None = None
_pixel_points: list[tuple[int, int]] = []
_world_points: list[tuple[float, float, float]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Camera-to-arm calibration using point correspondences.",
    )
    parser.add_argument("--config-dir", default="config", help="Config directory (default: config).")
    parser.add_argument(
        "--output", default="config/calibration_result.yaml",
        help="Output YAML file for the homography matrix (default: config/calibration_result.yaml).",
    )
    parser.add_argument("--mock", action="store_true", help="Use mock camera for testing.")
    return parser.parse_args()


def mouse_callback(event: int, x: int, y: int, flags: int, param) -> None:
    """Record a clicked pixel coordinate."""
    global _clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        _clicked_point = (x, y)


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw collected calibration points on the frame."""
    display = frame.copy()
    for i, (px, py) in enumerate(_pixel_points):
        cv2.circle(display, (px, py), 6, (0, 0, 255), -1)
        cv2.putText(display, str(i + 1), (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Draw the most recent click (not yet confirmed).
    if _clicked_point is not None:
        cv2.circle(display, _clicked_point, 8, (0, 255, 255), 2)

    n = len(_pixel_points)
    status = f"Points: {n} (need >= 4)  |  Click fruit, then enter arm XYZ in terminal"
    cv2.putText(display, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return display


def main() -> None:
    global _clicked_point

    args = parse_args()
    cfg = load_config(config_dir=args.config_dir)
    cam = CameraManager(cfg, mock_mode=args.mock)

    print("=== Camera-to-Arm Calibration ===")
    print()
    print("Instructions:")
    print("  1. Place a fruit at a known position on the belt.")
    print("  2. Click the fruit in the camera view.")
    print("  3. Enter the arm's XYZ coordinates in the terminal.")
    print("  4. Repeat for at least 4 points.")
    print("  5. Press 'q' in the camera window when done to compute homography.")
    print()

    window_name = "Calibration - Click fruit positions"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    cam.start()
    try:
        while True:
            frame = cam.get_overhead_frame()
            if frame is None:
                continue

            display = draw_overlay(frame)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break

            # When the user clicks a point, prompt for arm coordinates.
            if _clicked_point is not None:
                px, py = _clicked_point
                print(f"\nClicked pixel: ({px}, {py})")
                try:
                    coords_str = input("  Enter arm XYZ (comma-separated, e.g. 150.0,200.0,50.0): ").strip()
                    parts = [float(v.strip()) for v in coords_str.split(",")]
                    if len(parts) != 3:
                        print("  Error: expected 3 values (x, y, z). Discarding click.")
                        _clicked_point = None
                        continue
                    wx, wy, wz = parts
                except (ValueError, EOFError):
                    print("  Invalid input. Discarding click.")
                    _clicked_point = None
                    continue

                _pixel_points.append((px, py))
                _world_points.append((wx, wy, wz))
                print(f"  Point {len(_pixel_points)} saved: pixel=({px},{py}) -> arm=({wx},{wy},{wz})")
                _clicked_point = None

    finally:
        cam.stop()
        cv2.destroyAllWindows()

    # Compute homography.
    n = len(_pixel_points)
    if n < 4:
        print(f"\nError: need at least 4 point pairs, got {n}. Calibration aborted.")
        return

    print(f"\nComputing homography from {n} point pairs...")

    src_pts = np.array(_pixel_points, dtype=np.float64)
    # Use only XY from world coordinates for the 2D homography.
    dst_pts = np.array([(wx, wy) for wx, wy, _ in _world_points], dtype=np.float64)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("Error: homography computation failed.")
        return

    inliers = int(mask.sum()) if mask is not None else n
    print(f"Homography computed. Inliers: {inliers}/{n}")
    print()
    print("Homography matrix:")
    print(H)

    # Save to YAML.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    calibration_data = {
        "homography": H.tolist(),
        "num_points": n,
        "num_inliers": inliers,
        "pixel_points": [list(p) for p in _pixel_points],
        "world_points": [list(p) for p in _world_points],
    }

    with open(output_path, "w") as f:
        yaml.dump(calibration_data, f, default_flow_style=False)

    print(f"\nCalibration saved to: {output_path}")


if __name__ == "__main__":
    main()
