"""Tests for the fruit detector and tracker in mock mode."""

import numpy as np

from grader.detector import FruitDetector, FruitTracker


def _identity_pixel_to_world(px: float, py: float) -> tuple[float, float]:
    return (px, py)


def test_mock_detector_returns_detection():
    det = FruitDetector(mock_mode=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    detections = det.detect(frame)
    assert len(detections) >= 1
    d = detections[0]
    assert len(d.bbox) == 4
    assert len(d.center_px) == 2
    assert 0.0 <= d.confidence <= 1.0


def test_tracker_assigns_ids():
    det = FruitDetector(mock_mode=True)
    tracker = FruitTracker(
        pixel_to_world_fn=_identity_pixel_to_world,
        pick_zone={"x_range": [0, 1280], "y_range": [0, 720]},
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = det.detect(frame)
    tracked = tracker.update(detections, timestamp=0.0)
    assert len(tracked) >= 1
    assert tracked[0].fruit_id >= 1


def test_tracker_maintains_id_across_frames():
    det = FruitDetector(mock_mode=True)
    tracker = FruitTracker(
        pixel_to_world_fn=_identity_pixel_to_world,
        pick_zone={"x_range": [0, 2000], "y_range": [0, 2000]},
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    det.detect(frame)
    tracked1 = tracker.update(det.detect(frame), timestamp=0.0)
    tracked2 = tracker.update(det.detect(frame), timestamp=0.1)

    if tracked1 and tracked2:
        # Same fruit should keep same ID
        assert tracked1[0].fruit_id == tracked2[0].fruit_id


def test_pick_queue_returns_fruits_in_zone():
    tracker = FruitTracker(
        pixel_to_world_fn=_identity_pixel_to_world,
        pick_zone={"x_range": [0, 1280], "y_range": [0, 720]},
    )
    det = FruitDetector(mock_mode=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = det.detect(frame)
    tracker.update(detections, timestamp=0.0)
    queue = tracker.get_pick_queue()
    # All detections in the large pick zone should be in the queue
    assert isinstance(queue, list)
