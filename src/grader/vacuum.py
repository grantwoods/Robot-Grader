"""Vacuum solenoid controller for suction cup end effector."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class VacuumController:
    """GPIO-based vacuum solenoid control with timeout safety."""

    def __init__(self, config: dict, mock_mode: bool = False) -> None:
        self._mock = mock_mode
        self._gpio_pin: int = config.get("gpio_pin", 18)
        self._grip_delay: float = config.get("grip_confirm_delay_sec", 0.15)
        self._release_delay: float = config.get("release_confirm_delay_sec", 0.10)
        self._timeout: float = config.get("timeout_sec", 10.0)

        self._gripping: bool = False
        self._grip_start: float = 0.0
        self._timeout_timer: threading.Timer | None = None
        self._gpio = None

    def connect(self) -> None:
        if self._mock:
            logger.info("VacuumController: mock mode — no GPIO connected")
            return
        try:
            import Jetson.GPIO as GPIO  # type: ignore[import-untyped]

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._gpio_pin, GPIO.OUT, initial=GPIO.LOW)
            logger.info("VacuumController: GPIO pin %d configured", self._gpio_pin)
        except Exception:
            logger.exception("VacuumController: failed to set up GPIO")
            raise

    def disconnect(self) -> None:
        self.release()
        if self._gpio is not None:
            self._gpio.cleanup(self._gpio_pin)
            self._gpio = None
        logger.info("VacuumController: disconnected")

    def grip(self) -> None:
        if self._gripping:
            return
        if self._mock:
            logger.info("VacuumController: GRIP (mock)")
        else:
            self._gpio.output(self._gpio_pin, self._gpio.HIGH)
            logger.info("VacuumController: GRIP — pin %d HIGH", self._gpio_pin)
        self._gripping = True
        self._grip_start = time.time()
        time.sleep(self._grip_delay)
        self._start_timeout()

    def release(self) -> None:
        self._cancel_timeout()
        if not self._gripping:
            return
        if self._mock:
            logger.info("VacuumController: RELEASE (mock)")
        else:
            self._gpio.output(self._gpio_pin, self._gpio.LOW)
            logger.info("VacuumController: RELEASE — pin %d LOW", self._gpio_pin)
        self._gripping = False
        time.sleep(self._release_delay)

    def is_gripping(self) -> bool:
        return self._gripping

    def _start_timeout(self) -> None:
        self._cancel_timeout()
        self._timeout_timer = threading.Timer(self._timeout, self._on_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

    def _cancel_timeout(self) -> None:
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _on_timeout(self) -> None:
        logger.error("VacuumController: TIMEOUT — vacuum on for %.1fs, auto-releasing!", self._timeout)
        self.release()
