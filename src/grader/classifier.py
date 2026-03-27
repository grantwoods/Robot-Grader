"""Fruit visual classification and multi-view grade fusion."""

import enum
import logging
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Grade(enum.IntEnum):
    """Fruit quality grade, ordered worst to best."""

    TRASH = 0
    CHOICE = 1
    FANCY = 2


class FruitClassifier:
    """Classifies fruit quality from a single camera frame using a YOLO model."""

    def __init__(
        self,
        model_dir: str | Path,
        confidence_threshold: float = 0.5,
        mock_mode: bool = False,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._confidence_threshold = confidence_threshold
        self._mock_mode = mock_mode
        self._model = None
        self._fruit_type: str | None = None

    def load_model(self, fruit_type: str) -> None:
        self._fruit_type = fruit_type
        if self._mock_mode:
            logger.info("Mock mode: skipping model load for %s", fruit_type)
            return
        from ultralytics import YOLO

        model_path = self._model_dir / f"{fruit_type}_classifier.pt"
        logger.info("Loading classification model from %s", model_path)
        self._model = YOLO(str(model_path))

    def classify(self, frame: np.ndarray) -> tuple[Grade, float]:
        if self._mock_mode:
            grade = random.choice(list(Grade))
            confidence = random.uniform(0.5, 1.0)
            return grade, confidence

        results = self._model(frame, verbose=False)
        probs = results[0].probs
        class_idx = probs.top1
        confidence = float(probs.top1conf)

        if confidence < self._confidence_threshold:
            logger.debug(
                "Confidence %.3f below threshold %.3f, defaulting to CHOICE",
                confidence,
                self._confidence_threshold,
            )
            return Grade.CHOICE, confidence

        grade = Grade(class_idx)
        return grade, confidence


class GradeFusion:
    """Fuses grades from multiple camera views using downgrade-only logic."""

    def fuse(
        self,
        overhead: Grade,
        arm_head: Grade | None = None,
        bottom: Grade | None = None,
    ) -> Grade:
        grades = [overhead]
        if arm_head is not None:
            grades.append(arm_head)
        if bottom is not None:
            grades.append(bottom)
        return min(grades)

    def should_skip_bottom_inspect(
        self,
        overhead: Grade,
        arm_head: Grade | None = None,
    ) -> bool:
        if overhead is Grade.TRASH:
            return True
        if arm_head is Grade.TRASH:
            return True
        return False
