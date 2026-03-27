"""Camera manager for the robot fruit sorting system.

Manages three OpenCV VideoCapture instances (overhead, arm-mounted, bottom
inspection) and provides frame capture, preprocessing, and coordinate
transformation utilities.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraManager:
    """Manages overhead, arm-head, and bottom inspection cameras."""

    def __init__(self, config: dict, mock_mode: bool = False) -> None:
        self._mock_mode = mock_mode

        # Accept either the full config or just the cameras sub-dict
        cam_cfg = config.get("cameras", config) if "cameras" in config else config
        self._cameras_cfg = {
            "overhead": cam_cfg.get("overhead", {"device_index": 0, "resolution": [1280, 720], "fps": 30}),
            "arm_head": cam_cfg.get("arm_head", {"device_index": 1, "resolution": [1280, 720], "fps": 30}),
            "bottom": cam_cfg.get("bottom", {"device_index": 2, "resolution": [1280, 720], "fps": 30}),
        }
        self._model_input_size: tuple[int, int] = tuple(cam_cfg.get("model_input_size", [640, 640]))

        # Calibration can be nested under config or passed directly
        cal = config.get("calibration", {})
        homography_rows = cal.get("homography", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self._homography: np.ndarray = np.array(homography_rows, dtype=np.float64)

        self._caps: dict[str, cv2.VideoCapture | None] = {
            "overhead": None,
            "arm_head": None,
            "bottom": None,
        }

    # -- Context manager -------------------------------------------------------

    def __enter__(self) -> CameraManager:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # -- Lifecycle -------------------------------------------------------------

    def start(self) -> None:
        if self._mock_mode:
            logger.info("CameraManager starting in mock mode — no real cameras")
            return

        for name, cfg in self._cameras_cfg.items():
            cap = cv2.VideoCapture(cfg["device_index"])
            w, h = cfg["resolution"]
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, cfg["fps"])

            if not cap.isOpened():
                logger.error("Failed to open %s camera (device %d)", name, cfg["device_index"])
            else:
                logger.info("Opened %s camera (device %d, %dx%d @ %d fps)",
                            name, cfg["device_index"], w, h, cfg["fps"])

            self._caps[name] = cap

    def stop(self) -> None:
        for name, cap in self._caps.items():
            if cap is not None:
                cap.release()
                logger.info("Released %s camera", name)
                self._caps[name] = None

    # -- Frame capture ---------------------------------------------------------

    def _mock_frame(self, name: str) -> np.ndarray:
        w, h = self._cameras_cfg[name]["resolution"]
        colors = {
            "overhead": (200, 180, 60),   # blueish
            "arm_head": (60, 200, 80),    # greenish
            "bottom": (60, 80, 200),      # reddish
        }
        b, g, r = colors[name]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (b, g, r)
        return frame

    def _read_frame(self, name: str) -> np.ndarray | None:
        if self._mock_mode:
            return self._mock_frame(name)

        cap = self._caps.get(name)
        if cap is None or not cap.isOpened():
            logger.warning("Camera %s is not available", name)
            return None

        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from %s camera", name)
            return None
        return frame

    def get_overhead_frame(self) -> np.ndarray | None:
        return self._read_frame("overhead")

    def capture_arm_closeup(self) -> np.ndarray | None:
        return self._read_frame("arm_head")

    def capture_bottom_view(self) -> np.ndarray | None:
        return self._read_frame("bottom")

    # -- Processing ------------------------------------------------------------

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        target_w, target_h = self._model_input_size
        resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def pixel_to_world(self, px: float, py: float) -> tuple[float, float]:
        pt = np.array([px, py, 1.0], dtype=np.float64)
        world = self._homography @ pt
        world /= world[2]
        return (float(world[0]), float(world[1]))
