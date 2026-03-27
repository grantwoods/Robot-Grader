"""Structured logging for fruit sorting records."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FruitRecord:
    """One complete record of a fruit passing through the sort cycle."""

    timestamp: float = field(default_factory=time.time)
    fruit_id: int = 0
    overhead_grade: str = ""
    overhead_confidence: float = 0.0
    arm_grade: str = ""
    arm_confidence: float = 0.0
    bottom_grade: str = ""
    bottom_confidence: float = 0.0
    final_grade: str = ""
    chute: str = ""
    cycle_time_ms: float = 0.0
    pressure_at_contact: float = 0.0
    pick_success: bool = False

    _FIELDNAMES: list[str] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_FIELDNAMES",
            [f.name for f in self.__dataclass_fields__.values() if f.name != "_FIELDNAMES"],
        )


class SortLogger:
    """Appends :class:`FruitRecord` rows to a daily CSV file."""

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_path: Optional[Path] = None
        self._writer: Optional[csv.DictWriter] = None
        self._file = None

    def _ensure_file(self) -> None:
        today = time.strftime("%Y-%m-%d")
        path = self.log_dir / f"sort_log_{today}.csv"
        if path == self._current_path and self._file is not None:
            return
        self.close()
        is_new = not path.exists()
        self._file = open(path, "a", newline="")
        fieldnames = [
            f.name
            for f in FruitRecord.__dataclass_fields__.values()
            if f.name != "_FIELDNAMES"
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if is_new:
            self._writer.writeheader()
        self._current_path = path

    def log(self, record: FruitRecord) -> None:
        self._ensure_file()
        row = asdict(record)
        row.pop("_FIELDNAMES", None)
        assert self._writer is not None
        self._writer.writerow(row)
        assert self._file is not None
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            self._current_path = None
