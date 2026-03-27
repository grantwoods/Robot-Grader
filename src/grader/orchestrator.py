"""Main orchestrator — initializes all subsystems and runs the sort loop."""

from __future__ import annotations

import logging
import signal
import sys
import time

from grader.arm import ArmController
from grader.camera import CameraManager
from grader.classifier import FruitClassifier, GradeFusion
from grader.config import load_config
from grader.detector import FruitDetector, FruitTracker
from grader.interceptor import InterceptionCalculator
from grader.logging_utils import SortLogger
from grader.pressure import PressureSensor
from grader.state_machine import SortCycle
from grader.vacuum import VacuumController

logger = logging.getLogger(__name__)


class Orchestrator:
    """Ties all subsystems together and provides the operator interface."""

    def __init__(self, config: dict) -> None:
        self.config = config
        mock = config.get("system", {}).get("mock_mode", True)

        self.camera = CameraManager(config, mock_mode=mock)
        self.detector = FruitDetector(mock_mode=mock)
        self.tracker = FruitTracker(
            pixel_to_world_fn=self.camera.pixel_to_world,
            pick_zone=config.get("arm", {}).get("pick_zone", {}),
        )
        self.classifier = FruitClassifier(
            model_dir=config.get("classifier", {}).get("model_dir", "models"),
            confidence_threshold=config.get("classifier", {}).get("confidence_threshold", 0.5),
            mock_mode=mock,
        )
        self.fusion = GradeFusion()
        self.arm = ArmController(config.get("arm", {}), mock_mode=mock)
        self.interceptor = InterceptionCalculator(
            safe_z_height=config.get("arm", {}).get("safe_z_height", 150.0),
        )
        self.pressure = PressureSensor(config.get("pressure", {}), mock_mode=mock)
        self.vacuum = VacuumController(config.get("vacuum", {}), mock_mode=mock)
        self.sort_cycle = SortCycle(
            camera=self.camera,
            detector=self.detector,
            tracker=self.tracker,
            classifier=self.classifier,
            fusion=self.fusion,
            arm=self.arm,
            interceptor=self.interceptor,
            pressure=self.pressure,
            vacuum=self.vacuum,
            config=config,
        )
        self.sort_logger = SortLogger(config.get("system", {}).get("log_dir", "logs"))
        self._running = False
        self._fruit_count = 0

    def start(self) -> None:
        fruit_type = self.config.get("fruit_type", "lemon")
        logger.info("=== Wildwood Robot Grader ===")
        logger.info("Fruit type: %s", fruit_type)
        logger.info("Mock mode: %s", self.config.get("system", {}).get("mock_mode", True))
        logger.info("Shadow mode: %s", self.config.get("system", {}).get("shadow_mode", False))

        self.camera.start()
        self.arm.connect()
        self.pressure.connect()
        self.vacuum.connect()
        self.classifier.load_model(fruit_type)
        self.arm.home()

        logger.info("All subsystems initialized — ready to sort")

    def stop(self) -> None:
        self._running = False
        self.vacuum.release()
        self.arm.home()
        self.arm.disconnect()
        self.pressure.disconnect()
        self.vacuum.disconnect()
        self.camera.stop()
        self.sort_logger.close()
        logger.info("System stopped — %d fruit sorted this session", self._fruit_count)

    def run(self) -> None:
        self.start()
        self._running = True

        def _handle_signal(sig: int, frame: object) -> None:
            logger.info("Shutdown signal received")
            self._running = False

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        logger.info("Sort loop running — press Ctrl+C to stop")
        try:
            while self._running:
                record = self.sort_cycle.tick()
                if record is not None:
                    self.sort_logger.log(record)
                    self._fruit_count += 1
                    logger.info(
                        "Sorted: #%d -> %s (%s) [%.0fms]",
                        record.fruit_id,
                        record.chute,
                        record.final_grade,
                        record.cycle_time_ms,
                    )
                time.sleep(0.01)  # ~100 Hz main loop
        finally:
            self.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    fruit_type = None
    if len(sys.argv) > 1:
        fruit_type = sys.argv[1]

    config = load_config(fruit_type=fruit_type)
    orchestrator = Orchestrator(config)
    orchestrator.run()


if __name__ == "__main__":
    main()
