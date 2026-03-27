"""Fruit detection and tracking for the conveyor belt overhead camera.

Uses basic OpenCV background subtraction and contour detection (no ML models)
to find fruit, then tracks them across frames with nearest-neighbor matching.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]          # x1, y1, x2, y2
    center_px: tuple[float, float]
    confidence: float


@dataclass
class TrackedFruit:
    fruit_id: int
    position: tuple[float, float]            # world coords (mm)
    velocity: tuple[float, float]            # mm/sec
    last_seen: float                          # timestamp
    overhead_grade: str | None = None
    overhead_confidence: float = 0.0


class FruitDetector:
    """Detects fruit on the conveyor belt using background subtraction and contours."""

    _MIN_CONTOUR_AREA = 500
    _MAX_CONTOUR_AREA = 50000

    def __init__(self, mock_mode: bool) -> None:
        self._mock_mode = mock_mode
        self._bg_subtractor: cv2.BackgroundSubtractorMOG2 | None = None
        self._mock_tick: int = 0

        if not mock_mode:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=50,
                detectShadows=True,
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._mock_mode:
            return self._mock_detect(frame)
        return self._real_detect(frame)

    def _mock_detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        self._mock_tick += 1

        # Fruit moves across the frame horizontally over time
        cx = (self._mock_tick * 5) % w
        cy = h // 2
        half = 30

        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, w)
        y2 = min(cy + half, h)

        return [Detection(
            bbox=(x1, y1, x2, y2),
            center_px=(float(cx), float(cy)),
            confidence=0.95,
        )]

    def _real_detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        fg_mask = self._bg_subtractor.apply(blurred)

        # Remove shadows (shadow pixels are marked as 127 by MOG2)
        _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._MIN_CONTOUR_AREA or area > self._MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Confidence heuristic based on contour solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0.0

            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                center_px=(cx, cy),
                confidence=min(solidity, 1.0),
            ))

        return detections


class FruitTracker:
    """Tracks detected fruit across frames using nearest-neighbor matching."""

    _MAX_MATCH_DISTANCE = 80.0  # pixels

    def __init__(
        self,
        pixel_to_world_fn: Callable[[float, float], tuple[float, float]],
        pick_zone: dict,
        max_lost_frames: int = 10,
    ) -> None:
        self._pixel_to_world = pixel_to_world_fn
        self._pick_zone = pick_zone
        self._max_lost_frames = max_lost_frames

        self._next_id: int = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[Detection], timestamp: float) -> list[TrackedFruit]:
        # Build cost matrix: existing tracks vs new detections (pixel distance)
        track_ids = list(self._tracks.keys())
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        if track_ids and detections:
            track_centers = np.array([self._tracks[tid].last_center_px for tid in track_ids])
            det_centers = np.array([d.center_px for d in detections])

            # Pairwise distances
            diff = track_centers[:, np.newaxis, :] - det_centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)  # (n_tracks, n_dets)

            # Greedy nearest-neighbor matching
            while True:
                if dists.size == 0:
                    break
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                min_dist = dists[min_idx]
                if min_dist > self._MAX_MATCH_DISTANCE:
                    break

                ti, di = int(min_idx[0]), int(min_idx[1])
                tid = track_ids[ti]

                matched_track_ids.add(tid)
                matched_det_indices.add(di)

                det = detections[di]
                world_pos = self._pixel_to_world(*det.center_px)
                self._tracks[tid].update(world_pos, det.center_px, timestamp)

                # Invalidate row and column
                dists[ti, :] = np.inf
                dists[:, di] = np.inf

        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in matched_det_indices:
                continue
            world_pos = self._pixel_to_world(*det.center_px)
            track = _Track(
                fruit_id=self._next_id,
                position=world_pos,
                center_px=det.center_px,
                timestamp=timestamp,
            )
            self._tracks[self._next_id] = track
            self._next_id += 1

        # Increment lost counter for unmatched existing tracks
        for tid in track_ids:
            if tid not in matched_track_ids:
                self._tracks[tid].lost_frames += 1

        # Prune stale tracks
        stale = [tid for tid, t in self._tracks.items() if t.lost_frames > self._max_lost_frames]
        for tid in stale:
            del self._tracks[tid]

        # Build output
        results: list[TrackedFruit] = []
        for track in self._tracks.values():
            results.append(TrackedFruit(
                fruit_id=track.fruit_id,
                position=track.position,
                velocity=track.velocity,
                last_seen=track.last_timestamp,
                overhead_grade=track.overhead_grade,
                overhead_confidence=track.overhead_confidence,
            ))
        return results

    def get_pick_queue(self) -> list[TrackedFruit]:
        x_lo, x_hi = self._pick_zone["x_range"]
        y_lo, y_hi = self._pick_zone["y_range"]

        in_zone: list[TrackedFruit] = []
        for track in self._tracks.values():
            wx, wy = track.position
            if x_lo <= wx <= x_hi and y_lo <= wy <= y_hi:
                in_zone.append(TrackedFruit(
                    fruit_id=track.fruit_id,
                    position=track.position,
                    velocity=track.velocity,
                    last_seen=track.last_timestamp,
                    overhead_grade=track.overhead_grade,
                    overhead_confidence=track.overhead_confidence,
                ))

        # Order by proximity to exiting the pick zone (highest y = closest to exit)
        in_zone.sort(key=lambda f: f.position[1], reverse=True)
        return in_zone


class _Track:
    """Internal mutable tracking state for a single fruit."""

    def __init__(
        self,
        fruit_id: int,
        position: tuple[float, float],
        center_px: tuple[float, float],
        timestamp: float,
    ) -> None:
        self.fruit_id = fruit_id
        self.position = position
        self.last_center_px = center_px
        self.last_timestamp = timestamp
        self.velocity: tuple[float, float] = (0.0, 0.0)
        self.lost_frames: int = 0
        self.overhead_grade: str | None = None
        self.overhead_confidence: float = 0.0

    def update(
        self,
        world_pos: tuple[float, float],
        center_px: tuple[float, float],
        timestamp: float,
    ) -> None:
        dt = timestamp - self.last_timestamp
        if dt > 0:
            vx = (world_pos[0] - self.position[0]) / dt
            vy = (world_pos[1] - self.position[1]) / dt
            self.velocity = (vx, vy)

        self.position = world_pos
        self.last_center_px = center_px
        self.last_timestamp = timestamp
        self.lost_frames = 0
