"""Tests for the visual classifier in mock mode."""

import numpy as np

from grader.classifier import Grade, FruitClassifier


def test_mock_classify_returns_valid_grade():
    clf = FruitClassifier(model_dir="models", mock_mode=True)
    clf.load_model("lemon")
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    grade, conf = clf.classify(frame)
    assert isinstance(grade, Grade)
    assert 0.0 <= conf <= 1.0


def test_mock_load_model():
    clf = FruitClassifier(model_dir="models", mock_mode=True)
    clf.load_model("orange")
    assert clf._fruit_type == "orange"


def test_grade_ordering():
    assert Grade.TRASH < Grade.CHOICE < Grade.FANCY
    assert min(Grade.FANCY, Grade.TRASH) == Grade.TRASH
    assert min(Grade.CHOICE, Grade.FANCY) == Grade.CHOICE
