"""Exhaustive tests for the three-view grade fusion logic."""

from grader.classifier import Grade, GradeFusion


def test_all_fancy():
    f = GradeFusion()
    assert f.fuse(Grade.FANCY, Grade.FANCY, Grade.FANCY) == Grade.FANCY


def test_all_trash():
    f = GradeFusion()
    assert f.fuse(Grade.TRASH, Grade.TRASH, Grade.TRASH) == Grade.TRASH


def test_all_choice():
    f = GradeFusion()
    assert f.fuse(Grade.CHOICE, Grade.CHOICE, Grade.CHOICE) == Grade.CHOICE


def test_one_trash_overrides_fancy():
    f = GradeFusion()
    assert f.fuse(Grade.FANCY, Grade.FANCY, Grade.TRASH) == Grade.TRASH
    assert f.fuse(Grade.FANCY, Grade.TRASH, Grade.FANCY) == Grade.TRASH
    assert f.fuse(Grade.TRASH, Grade.FANCY, Grade.FANCY) == Grade.TRASH


def test_one_choice_downgrades_fancy():
    f = GradeFusion()
    assert f.fuse(Grade.FANCY, Grade.FANCY, Grade.CHOICE) == Grade.CHOICE
    assert f.fuse(Grade.FANCY, Grade.CHOICE, Grade.FANCY) == Grade.CHOICE
    assert f.fuse(Grade.CHOICE, Grade.FANCY, Grade.FANCY) == Grade.CHOICE


def test_trash_beats_choice():
    f = GradeFusion()
    assert f.fuse(Grade.CHOICE, Grade.CHOICE, Grade.TRASH) == Grade.TRASH
    assert f.fuse(Grade.TRASH, Grade.CHOICE, Grade.CHOICE) == Grade.TRASH


def test_two_views_only():
    f = GradeFusion()
    assert f.fuse(Grade.FANCY, Grade.FANCY) == Grade.FANCY
    assert f.fuse(Grade.FANCY, Grade.TRASH) == Grade.TRASH
    assert f.fuse(Grade.CHOICE, Grade.FANCY) == Grade.CHOICE


def test_overhead_only():
    f = GradeFusion()
    assert f.fuse(Grade.FANCY) == Grade.FANCY
    assert f.fuse(Grade.TRASH) == Grade.TRASH
    assert f.fuse(Grade.CHOICE) == Grade.CHOICE


def test_skip_bottom_inspect_trash_overhead():
    f = GradeFusion()
    assert f.should_skip_bottom_inspect(Grade.TRASH) is True
    assert f.should_skip_bottom_inspect(Grade.TRASH, Grade.FANCY) is True


def test_skip_bottom_inspect_trash_arm():
    f = GradeFusion()
    assert f.should_skip_bottom_inspect(Grade.FANCY, Grade.TRASH) is True
    assert f.should_skip_bottom_inspect(Grade.CHOICE, Grade.TRASH) is True


def test_no_skip_for_good_fruit():
    f = GradeFusion()
    assert f.should_skip_bottom_inspect(Grade.FANCY) is False
    assert f.should_skip_bottom_inspect(Grade.FANCY, Grade.FANCY) is False
    assert f.should_skip_bottom_inspect(Grade.CHOICE, Grade.CHOICE) is False
    assert f.should_skip_bottom_inspect(Grade.FANCY, Grade.CHOICE) is False


def test_exhaustive_three_view_combinations():
    """Verify the downgrade-only rule for all 27 three-view combinations."""
    f = GradeFusion()
    grades = [Grade.TRASH, Grade.CHOICE, Grade.FANCY]
    for a in grades:
        for b in grades:
            for c in grades:
                result = f.fuse(a, b, c)
                expected = min(a, b, c)
                assert result == expected, f"fuse({a}, {b}, {c}) = {result}, expected {expected}"
