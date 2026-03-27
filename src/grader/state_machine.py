"""Sort cycle state machine — the core control loop for fruit sorting."""

from __future__ import annotations

import enum
import logging
import time

from grader.camera import CameraManager
from grader.classifier import Grade, FruitClassifier, GradeFusion
from grader.detector import FruitDetector, FruitTracker, TrackedFruit
from grader.arm import ArmController
from grader.interceptor import InterceptionCalculator
from grader.logging_utils import FruitRecord
from grader.pressure import PressureSensor
from grader.vacuum import VacuumController

logger = logging.getLogger(__name__)

CHUTE_MAP = {
    Grade.TRASH: "chute_a",
    Grade.CHOICE: "chute_b",
    Grade.FANCY: "chute_c",
}


class SortState(enum.Enum):
    WAITING = "WAITING"
    INTERCEPTING = "INTERCEPTING"
    ARM_CAMERA_CLOSE_UP = "ARM_CAMERA_CLOSE_UP"
    CONTACT_AND_PICK = "CONTACT_AND_PICK"
    INITIAL_CLASSIFY = "INITIAL_CLASSIFY"
    BOTTOM_INSPECT = "BOTTOM_INSPECT"
    SORTING = "SORTING"
    DROPPING = "DROPPING"


class SortCycle:
    """Implements the full fruit-sorting state machine.

    Call ``tick()`` repeatedly from the main loop. Each call evaluates the
    current state, performs the appropriate action, and transitions when ready.
    """

    def __init__(
        self,
        camera: CameraManager,
        detector: FruitDetector,
        tracker: FruitTracker,
        classifier: FruitClassifier,
        fusion: GradeFusion,
        arm: ArmController,
        interceptor: InterceptionCalculator,
        pressure: PressureSensor,
        vacuum: VacuumController,
        config: dict,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.classifier = classifier
        self.fusion = fusion
        self.arm = arm
        self.interceptor = interceptor
        self.pressure = pressure
        self.vacuum = vacuum

        self._shadow_mode: bool = config.get("system", {}).get("shadow_mode", False)
        pick_zone = config.get("arm", {}).get("pick_zone", {})
        self._pick_zone_x = tuple(pick_zone.get("x_range", [-100, 100]))
        self._pick_zone_y = tuple(pick_zone.get("y_range", [150, 250]))

        self._state = SortState.WAITING
        self._current_fruit: TrackedFruit | None = None
        self._record = FruitRecord()
        self._cycle_start: float = 0.0

        # Classification results accumulated across views
        self._overhead_grade: Grade | None = None
        self._overhead_conf: float = 0.0
        self._arm_grade: Grade | None = None
        self._arm_conf: float = 0.0
        self._bottom_grade: Grade | None = None
        self._bottom_conf: float = 0.0
        self._final_grade: Grade | None = None

        # Contact detection flag for async pressure monitoring
        self._contact_detected = False

    @property
    def state(self) -> SortState:
        return self._state

    def tick(self, timestamp: float | None = None) -> FruitRecord | None:
        """Advance the state machine by one step. Returns a FruitRecord when a
        fruit completes the full cycle (or fails), otherwise None."""
        ts = timestamp if timestamp is not None else time.time()
        handler = {
            SortState.WAITING: self._tick_waiting,
            SortState.INTERCEPTING: self._tick_intercepting,
            SortState.ARM_CAMERA_CLOSE_UP: self._tick_arm_camera,
            SortState.CONTACT_AND_PICK: self._tick_contact_and_pick,
            SortState.INITIAL_CLASSIFY: self._tick_initial_classify,
            SortState.BOTTOM_INSPECT: self._tick_bottom_inspect,
            SortState.SORTING: self._tick_sorting,
            SortState.DROPPING: self._tick_dropping,
        }
        return handler[self._state](ts)

    # -- State handlers -------------------------------------------------------

    def _transition(self, new_state: SortState) -> None:
        logger.info("State: %s -> %s", self._state.value, new_state.value)
        self._state = new_state

    def _tick_waiting(self, ts: float) -> FruitRecord | None:
        frame = self.camera.get_overhead_frame()
        if frame is None:
            return None
        detections = self.detector.detect(frame)
        self.tracker.update(detections, ts)
        queue = self.tracker.get_pick_queue()
        if not queue:
            return None

        fruit = queue[0]
        self._current_fruit = fruit
        self._cycle_start = ts
        self._record = FruitRecord(timestamp=ts, fruit_id=fruit.fruit_id)

        # Run overhead classification on the detection
        overhead_frame = self.camera.preprocess(frame)
        grade, conf = self.classifier.classify(overhead_frame)
        self._overhead_grade = grade
        self._overhead_conf = conf
        self._record.overhead_grade = grade.name
        self._record.overhead_confidence = conf

        # Store on tracked fruit for reference
        fruit.overhead_grade = grade.name
        fruit.overhead_confidence = conf

        logger.info(
            "Fruit #%d detected — overhead grade: %s (%.2f)",
            fruit.fruit_id, grade.name, conf,
        )
        self._transition(SortState.INTERCEPTING)
        return None

    def _tick_intercepting(self, ts: float) -> FruitRecord | None:
        fruit = self._current_fruit
        assert fruit is not None

        intercept = self.interceptor.calculate_intercept(
            fruit_pos=fruit.position,
            fruit_velocity=fruit.velocity,
            arm_travel_time_fn=self.arm.estimate_travel_time,
            pick_zone_x_range=self._pick_zone_x,
            pick_zone_y_range=self._pick_zone_y,
        )
        if intercept is None:
            logger.warning("Fruit #%d — cannot intercept, skipping", fruit.fruit_id)
            return self._abort_cycle("intercept_failed")

        if not self._shadow_mode:
            self.arm.move_to(*intercept)
        self._transition(SortState.ARM_CAMERA_CLOSE_UP)
        return None

    def _tick_arm_camera(self, ts: float) -> FruitRecord | None:
        # Capture close-up from the arm-mounted camera during approach
        frame = self.camera.capture_arm_closeup()
        if frame is not None:
            processed = self.camera.preprocess(frame)
            grade, conf = self.classifier.classify(processed)
            self._arm_grade = grade
            self._arm_conf = conf
            self._record.arm_grade = grade.name
            self._record.arm_confidence = conf
            logger.info("Fruit #%d — arm camera grade: %s (%.2f)",
                        self._record.fruit_id, grade.name, conf)

        # Check if we can fast-track to trash
        if self._overhead_grade == Grade.TRASH or self._arm_grade == Grade.TRASH:
            logger.info("Fruit #%d — trash detected early, fast-tracking", self._record.fruit_id)

        self._transition(SortState.CONTACT_AND_PICK)
        return None

    def _tick_contact_and_pick(self, ts: float) -> FruitRecord | None:
        if self._shadow_mode:
            self._record.pick_success = True
            self._transition(SortState.INITIAL_CLASSIFY)
            return None

        fruit = self._current_fruit
        assert fruit is not None

        # Pressure-guided descent
        self._contact_detected = False
        self.pressure.reset_mock()

        safe_z = self.arm._safe_z_height
        contact_z = self.arm.descend_to_contact(
            target_x=fruit.position[0],
            target_y=fruit.position[1],
            start_z=safe_z,
            pressure_callback=self.pressure.is_contact,
        )
        self._record.pressure_at_contact = self.pressure.get_contact_force()

        # Activate vacuum
        self.vacuum.grip()
        self._record.pick_success = True
        logger.info("Fruit #%d — picked at Z=%.1f, pressure=%.3f",
                     self._record.fruit_id, contact_z, self._record.pressure_at_contact)
        self._transition(SortState.INITIAL_CLASSIFY)
        return None

    def _tick_initial_classify(self, ts: float) -> FruitRecord | None:
        # Decide whether to inspect the bottom
        skip = self.fusion.should_skip_bottom_inspect(
            self._overhead_grade or Grade.CHOICE,
            self._arm_grade,
        )
        if skip:
            # Final grade from available views only
            self._final_grade = self.fusion.fuse(
                self._overhead_grade or Grade.CHOICE,
                self._arm_grade,
            )
            self._record.final_grade = self._final_grade.name
            self._transition(SortState.SORTING)
        else:
            self._transition(SortState.BOTTOM_INSPECT)
        return None

    def _tick_bottom_inspect(self, ts: float) -> FruitRecord | None:
        if not self._shadow_mode:
            self.arm.move_to_waypoint("inspect")

        frame = self.camera.capture_bottom_view()
        if frame is not None:
            processed = self.camera.preprocess(frame)
            grade, conf = self.classifier.classify(processed)
            self._bottom_grade = grade
            self._bottom_conf = conf
            self._record.bottom_grade = grade.name
            self._record.bottom_confidence = conf
            logger.info("Fruit #%d — bottom grade: %s (%.2f)",
                        self._record.fruit_id, grade.name, conf)

        self._final_grade = self.fusion.fuse(
            self._overhead_grade or Grade.CHOICE,
            self._arm_grade,
            self._bottom_grade,
        )
        self._record.final_grade = self._final_grade.name
        logger.info("Fruit #%d — final grade: %s", self._record.fruit_id, self._final_grade.name)
        self._transition(SortState.SORTING)
        return None

    def _tick_sorting(self, ts: float) -> FruitRecord | None:
        assert self._final_grade is not None
        chute = CHUTE_MAP[self._final_grade]
        self._record.chute = chute
        if not self._shadow_mode:
            self.arm.move_to_waypoint(chute)
        logger.info("Fruit #%d -> %s (%s)", self._record.fruit_id, chute, self._final_grade.name)
        self._transition(SortState.DROPPING)
        return None

    def _tick_dropping(self, ts: float) -> FruitRecord | None:
        if not self._shadow_mode:
            self.vacuum.release()
            self.arm.home()

        self._record.cycle_time_ms = (ts - self._cycle_start) * 1000
        record = self._record
        logger.info(
            "Fruit #%d — cycle complete in %.0fms",
            record.fruit_id, record.cycle_time_ms,
        )
        self._reset()
        return record

    # -- Helpers ---------------------------------------------------------------

    def _abort_cycle(self, reason: str) -> FruitRecord:
        self._record.pick_success = False
        self._record.final_grade = reason
        record = self._record
        logger.warning("Fruit #%d — cycle aborted: %s", record.fruit_id, reason)
        if not self._shadow_mode:
            self.vacuum.release()
            self.arm.home()
        self._reset()
        return record

    def _reset(self) -> None:
        self._state = SortState.WAITING
        self._current_fruit = None
        self._overhead_grade = None
        self._overhead_conf = 0.0
        self._arm_grade = None
        self._arm_conf = 0.0
        self._bottom_grade = None
        self._bottom_conf = 0.0
        self._final_grade = None
        self._contact_detected = False
