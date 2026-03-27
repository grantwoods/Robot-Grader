#!/usr/bin/env python3
"""Collect training data by recording camera frames while a human grader sorts fruit.

Usage:
    python scripts/collect_training_data.py --fruit-type apple
    python scripts/collect_training_data.py --fruit-type apple --mock
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2

from grader.camera import CameraManager
from grader.config import load_config
from grader.detector import FruitDetector

GRADE_KEYS = {
    ord("t"): "trash",
    ord("c"): "choice",
    ord("f"): "fancy",
}

CAMERA_ANGLES = ["overhead", "arm_closeup", "bottom"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect training data for fruit grading.",
    )
    parser.add_argument("--fruit-type", required=True, help="Type of fruit being graded.")
    parser.add_argument("--output-dir", default="data", help="Root directory for saved images (default: data).")
    parser.add_argument("--config-dir", default="config", help="Config directory (default: config).")
    parser.add_argument("--mock", action="store_true", help="Use mock cameras for testing.")
    return parser.parse_args()


def save_crop(frame, bbox, path: Path) -> None:
    """Crop the bounding box region from *frame* and save as JPEG."""
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size > 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), crop)


def save_full_frame(frame, path: Path) -> None:
    """Save a full frame as JPEG."""
    if frame is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)


def draw_detections(frame, detections):
    """Draw bounding boxes on a copy of the frame and return it."""
    display = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.confidence:.0%}"
        cv2.putText(display, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return display


def main() -> None:
    args = parse_args()

    # Real mode by default for data collection; --mock overrides for testing.
    mock_mode = args.mock

    cfg = load_config(config_dir=args.config_dir, fruit_type=args.fruit_type)
    cam = CameraManager(cfg, mock_mode=mock_mode)
    detector = FruitDetector(mock_mode=mock_mode)

    output_dir = Path(args.output_dir)
    counts: dict[str, int] = {"trash": 0, "choice": 0, "fancy": 0}

    print("=== Training Data Collection ===")
    print(f"Fruit type : {args.fruit_type}")
    print(f"Output dir : {output_dir}")
    print(f"Mock mode  : {mock_mode}")
    print()
    print("Controls:")
    print("  t = trash   c = choice   f = fancy   q = quit")
    print()

    cam.start()
    try:
        while True:
            # Capture overhead frame and detect fruit.
            overhead_frame = cam.get_overhead_frame()
            if overhead_frame is None:
                print("Warning: no overhead frame available.")
                time.sleep(0.1)
                continue

            detections = detector.detect(overhead_frame)
            display = draw_detections(overhead_frame, detections)

            # Show count overlay.
            info = "  ".join(f"{k}: {v}" for k, v in counts.items())
            cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Overhead - Training Data Collection", display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                print("\nQuitting.")
                break

            if key in GRADE_KEYS and detections:
                grade = GRADE_KEYS[key]
                ts = int(time.time() * 1000)

                # Save cropped overhead detections.
                for i, det in enumerate(detections):
                    fname = f"{ts}_{i}.jpg"
                    path = output_dir / args.fruit_type / grade / "overhead" / fname
                    save_crop(overhead_frame, det.bbox, path)

                # Capture and save arm_closeup and bottom frames (operator positions fruit).
                arm_frame = cam.capture_arm_closeup()
                if arm_frame is not None:
                    fname = f"{ts}.jpg"
                    path = output_dir / args.fruit_type / grade / "arm_closeup" / fname
                    save_full_frame(arm_frame, path)

                bottom_frame = cam.capture_bottom_view()
                if bottom_frame is not None:
                    fname = f"{ts}.jpg"
                    path = output_dir / args.fruit_type / grade / "bottom" / fname
                    save_full_frame(bottom_frame, path)

                counts[grade] += 1
                print(f"  Saved as '{grade}' — totals: {counts}")

    finally:
        cam.stop()
        cv2.destroyAllWindows()

    print("\nFinal counts:", counts)
    total = sum(counts.values())
    print(f"Total samples: {total}")


if __name__ == "__main__":
    main()
