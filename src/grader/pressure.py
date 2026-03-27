"""Pressure sensor controller for contact detection during arm descent."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class PressureSensor:
    """Reads a force-sensitive resistor via ADC and detects fruit contact."""

    def __init__(
        self,
        config: dict,
        mock_mode: bool = False,
    ) -> None:
        self._mock = mock_mode
        self._adc_address: int = config.get("adc_address", 0x48)
        self._adc_channel: int = config.get("adc_channel", 0)
        self._contact_threshold: float = config.get("contact_threshold", 0.15)
        self._max_force_threshold: float = config.get("max_force_threshold", 0.85)
        self._sample_rate_hz: int = config.get("sample_rate_hz", 100)

        self._adc = None
        self._last_reading: float = 0.0
        self._monitoring: bool = False
        self._monitor_thread: Optional[threading.Thread] = None

        # Mock state
        self._mock_step: int = 0

    def connect(self) -> None:
        if self._mock:
            logger.info("PressureSensor: mock mode — no ADC connected")
            return
        try:
            import board  # type: ignore[import-untyped]
            import busio  # type: ignore[import-untyped]
            import adafruit_ads1x15.ads1115 as ADS  # type: ignore[import-untyped]
            from adafruit_ads1x15.analog_in import AnalogIn  # type: ignore[import-untyped]

            i2c = busio.I2C(board.SCL, board.SDA)
            ads = ADS.ADS1115(i2c, address=self._adc_address)
            channels = [ADS.P0, ADS.P1, ADS.P2, ADS.P3]
            self._adc = AnalogIn(ads, channels[self._adc_channel])
            logger.info("PressureSensor: ADC connected at 0x%02X ch%d", self._adc_address, self._adc_channel)
        except Exception:
            logger.exception("PressureSensor: failed to connect ADC")
            raise

    def disconnect(self) -> None:
        self.stop_monitoring()
        self._adc = None
        logger.info("PressureSensor: disconnected")

    def read_pressure(self) -> float:
        if self._mock:
            self._mock_step += 1
            self._last_reading = min(1.0, self._mock_step * 0.05)
            return self._last_reading
        if self._adc is None:
            return 0.0
        raw = self._adc.voltage
        self._last_reading = max(0.0, min(1.0, raw / 3.3))
        return self._last_reading

    def is_contact(self, threshold: float | None = None) -> bool:
        t = threshold if threshold is not None else self._contact_threshold
        return self.read_pressure() >= t

    def is_over_max_force(self) -> bool:
        return self._last_reading >= self._max_force_threshold

    def reset_mock(self) -> None:
        self._mock_step = 0
        self._last_reading = 0.0

    def monitor_descent(
        self,
        on_contact: Callable[[], None],
        on_max_force: Callable[[], None] | None = None,
    ) -> None:
        self._monitoring = True
        interval = 1.0 / self._sample_rate_hz

        def _loop() -> None:
            while self._monitoring:
                pressure = self.read_pressure()
                if pressure >= self._max_force_threshold:
                    logger.warning("PressureSensor: MAX FORCE exceeded (%.3f)", pressure)
                    if on_max_force:
                        on_max_force()
                    self._monitoring = False
                    return
                if pressure >= self._contact_threshold:
                    logger.info("PressureSensor: contact detected (%.3f)", pressure)
                    on_contact()
                    self._monitoring = False
                    return
                time.sleep(interval)

        self._monitor_thread = threading.Thread(target=_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        self._monitoring = False
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None

    def get_contact_force(self) -> float:
        return self._last_reading
