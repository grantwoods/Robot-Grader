"""Tests for the sort cycle state machine in mock mode."""

from grader.arm import ArmController
from grader.camera import CameraManager
from grader.classifier import FruitClassifier, GradeFusion
from grader.config import load_config
from grader.detector import FruitDetector, FruitTracker
from grader.interceptor import InterceptionCalculator
from grader.pressure import PressureSensor
from grader.state_machine import SortCycle, SortState
from grader.vacuum import VacuumController


def _make_cycle() -> SortCycle:
    config = load_config()
    # Force mock mode
    config.setdefault("system", {})["mock_mode"] = True

    # Override pick zone to match mock detector's pixel-space coordinates
    # Mock detector produces fruit at cy=360 (half of 720) moving across x 0-1280
    config["arm"]["pick_zone"] = {
        "x_range": [0, 1280],
        "y_range": [0, 720],
    }
    # Override waypoints/bounds to work in pixel-space coordinates
    config["arm"]["workspace_bounds"] = {
        "x": [0, 1280],
        "y": [0, 720],
        "z": [0, 400],
    }
    config["arm"]["waypoints"] = {
        "home": [640, 360, 150],
        "inspect": [640, 360, 120],
        "chute_a": [200, 360, 100],
        "chute_b": [640, 360, 100],
        "chute_c": [1000, 360, 100],
    }

    camera = CameraManager(config, mock_mode=True)
    detector = FruitDetector(mock_mode=True)
    tracker = FruitTracker(
        pixel_to_world_fn=camera.pixel_to_world,
        pick_zone=config["arm"]["pick_zone"],
    )
    classifier = FruitClassifier(model_dir="models", mock_mode=True)
    classifier.load_model("lemon")
    fusion = GradeFusion()
    arm = ArmController(config.get("arm", {}), mock_mode=True)
    interceptor = InterceptionCalculator(
        safe_z_height=config.get("arm", {}).get("safe_z_height", 150.0),
    )
    pressure = PressureSensor(config.get("pressure", {}), mock_mode=True)
    vacuum = VacuumController(config.get("vacuum", {}), mock_mode=True)

    camera.start()
    arm.connect()
    pressure.connect()
    vacuum.connect()

    return SortCycle(
        camera=camera,
        detector=detector,
        tracker=tracker,
        classifier=classifier,
        fusion=fusion,
        arm=arm,
        interceptor=interceptor,
        pressure=pressure,
        vacuum=vacuum,
        config=config,
    )


def test_initial_state_is_waiting():
    cycle = _make_cycle()
    assert cycle.state == SortState.WAITING


def test_full_cycle_completes():
    """Run the state machine through a complete fruit sorting cycle in mock mode."""
    cycle = _make_cycle()
    record = None
    # Tick enough times to complete one full cycle
    for i in range(50):
        result = cycle.tick(timestamp=float(i) * 0.1)
        if result is not None:
            record = result
            break

    assert record is not None, "State machine did not complete a cycle within 50 ticks"
    assert record.fruit_id >= 1
    assert record.overhead_grade != ""
    assert record.final_grade != ""
    assert record.cycle_time_ms > 0


def test_cycle_returns_to_waiting():
    """After completing a cycle, the state machine should return to WAITING."""
    cycle = _make_cycle()
    for i in range(50):
        result = cycle.tick(timestamp=float(i) * 0.1)
        if result is not None:
            break
    assert cycle.state == SortState.WAITING
