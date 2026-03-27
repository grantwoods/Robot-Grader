"""Tests for the interception calculator."""

from grader.interceptor import InterceptionCalculator


def _constant_travel_time(x: float, y: float, z: float) -> float:
    return 0.5  # 500ms to reach any point


def test_stationary_fruit():
    calc = InterceptionCalculator(safe_z_height=150.0)
    result = calc.calculate_intercept(
        fruit_pos=(10.0, 200.0),
        fruit_velocity=(0.0, 0.0),
        arm_travel_time_fn=_constant_travel_time,
        pick_zone_x_range=(-100.0, 100.0),
        pick_zone_y_range=(150.0, 250.0),
    )
    assert result is not None
    assert result == (10.0, 200.0, 150.0)


def test_moving_fruit_intercept():
    calc = InterceptionCalculator(safe_z_height=150.0)
    result = calc.calculate_intercept(
        fruit_pos=(0.0, 160.0),
        fruit_velocity=(20.0, 0.0),  # 20 mm/sec in X
        arm_travel_time_fn=_constant_travel_time,
        pick_zone_x_range=(-100.0, 100.0),
        pick_zone_y_range=(150.0, 250.0),
    )
    assert result is not None
    x, y, z = result
    # After 0.5s at 20mm/s, fruit moves 10mm in X
    assert abs(x - 10.0) < 1.0
    assert abs(y - 160.0) < 1.0
    assert z == 150.0


def test_fruit_exits_pick_zone():
    calc = InterceptionCalculator(safe_z_height=150.0)
    result = calc.calculate_intercept(
        fruit_pos=(90.0, 200.0),
        fruit_velocity=(100.0, 0.0),  # fast, will exit zone
        arm_travel_time_fn=_constant_travel_time,
        pick_zone_x_range=(-100.0, 100.0),
        pick_zone_y_range=(150.0, 250.0),
    )
    assert result is None


def test_in_zone_check():
    calc = InterceptionCalculator()
    assert calc._in_zone(0, 200, (-100, 100), (150, 250)) is True
    assert calc._in_zone(101, 200, (-100, 100), (150, 250)) is False
    assert calc._in_zone(0, 300, (-100, 100), (150, 250)) is False
