"""Decayed progress ETA with pause-aware elapsed time and cached unit rates.

Estimates use an exponentially decayed average of instantaneous rate
(half-life default 20 minutes), so recent progress dominates. When a task
type completes, median seconds-per-unit is cached for the next run's
initial prior (then decayed away by live samples).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_HALF_LIFE_SEC = 20.0 * 60.0


def _decay_weight(age_sec: float, half_life_sec: float) -> float:
    if half_life_sec <= 0:
        return 1.0 if age_sec <= 0 else 0.0
    # w = 0.5 ** (age / half_life)
    return float(0.5 ** (age_sec / half_life_sec))


@dataclass
class ProgressSample:
    """One observation: completed units and active (non-paused) elapsed seconds."""

    units_done: float
    active_elapsed_sec: float
    wall_ts: float  # time.time() when observed


@dataclass
class EtaTracker:
    """Track progress for one task instance."""

    task_type: str
    total_units: float
    half_life_sec: float = DEFAULT_HALF_LIFE_SEC
    cache_path: Path | None = None
    units_done: float = 0.0
    active_elapsed_sec: float = 0.0
    _running: bool = False
    _segment_start_wall: float | None = None
    _samples: list[ProgressSample] = field(default_factory=list)
    _prior_sec_per_unit: float | None = None

    def __post_init__(self) -> None:
        if self.cache_path is not None:
            self._prior_sec_per_unit = load_cached_sec_per_unit(
                self.cache_path, self.task_type
            )

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._segment_start_wall = time.time()

    def pause(self) -> None:
        if not self._running:
            return
        now = time.time()
        if self._segment_start_wall is not None:
            self.active_elapsed_sec += max(0.0, now - self._segment_start_wall)
        self._segment_start_wall = None
        self._running = False

    def resume(self) -> None:
        self.start()

    def _flush_active(self) -> float:
        """Return active elapsed including current running segment."""
        elapsed = self.active_elapsed_sec
        if self._running and self._segment_start_wall is not None:
            elapsed += max(0.0, time.time() - self._segment_start_wall)
        return elapsed

    def record(self, units_done: float) -> dict[str, Any]:
        """Update completed units; returns current estimate snapshot."""
        self.units_done = float(units_done)
        active = self._flush_active()
        self.active_elapsed_sec = active if not self._running else self.active_elapsed_sec
        # Keep wall clock sample for decay ages
        self._samples.append(
            ProgressSample(
                units_done=self.units_done,
                active_elapsed_sec=self._flush_active(),
                wall_ts=time.time(),
            )
        )
        # Cap history
        if len(self._samples) > 500:
            self._samples = self._samples[-500:]
        return self.estimate()

    def mark_complete(self) -> float | None:
        """Finalize; cache sec/unit for this task_type. Returns sec/unit used."""
        self.pause()
        if self.units_done <= 0 or self.active_elapsed_sec <= 0:
            return None
        spu = self.active_elapsed_sec / self.units_done
        if self.cache_path is not None:
            save_cached_sec_per_unit(self.cache_path, self.task_type, spu)
        return spu

    def instantaneous_sec_per_unit(self) -> float | None:
        """Decayed average of interval rates between consecutive samples."""
        now = time.time()
        if len(self._samples) < 2:
            return None
        num = 0.0
        den = 0.0
        for prev, cur in zip(self._samples, self._samples[1:]):
            du = cur.units_done - prev.units_done
            dt = cur.active_elapsed_sec - prev.active_elapsed_sec
            if du <= 0 or dt <= 0:
                continue
            rate_spu = dt / du
            age = max(0.0, now - cur.wall_ts)
            w = _decay_weight(age, self.half_life_sec)
            num += w * rate_spu
            den += w
        if den <= 0:
            return None
        return num / den

    def blended_sec_per_unit(self) -> float | None:
        live = self.instantaneous_sec_per_unit()
        prior = self._prior_sec_per_unit
        if live is None and prior is None:
            # Fallback: overall active average so far
            active = self._flush_active()
            if self.units_done > 0 and active > 0:
                return active / self.units_done
            return None
        if live is None:
            return prior
        if prior is None:
            return live
        # Prior weight decays with total active time (same half-life): after
        # one half-life of active work, prior and live are equal; later live wins.
        active = self._flush_active()
        w_prior = _decay_weight(active, self.half_life_sec)
        return w_prior * prior + (1.0 - w_prior) * live

    def estimate(self) -> dict[str, Any]:
        remaining = max(0.0, self.total_units - self.units_done)
        spu = self.blended_sec_per_unit()
        eta_sec = (remaining * spu) if spu is not None else None
        active = self._flush_active()
        return {
            "task_type": self.task_type,
            "units_done": self.units_done,
            "total_units": self.total_units,
            "remaining_units": remaining,
            "active_elapsed_sec": active,
            "paused": not self._running,
            "sec_per_unit": spu,
            "eta_sec": eta_sec,
            "pct": (
                100.0 * self.units_done / self.total_units
                if self.total_units > 0
                else 0.0
            ),
            "prior_sec_per_unit": self._prior_sec_per_unit,
            "half_life_sec": self.half_life_sec,
        }


def load_cached_sec_per_unit(path: Path, task_type: str) -> float | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    entry = data.get(task_type)
    if not isinstance(entry, dict):
        return None
    try:
        v = float(entry["sec_per_unit"])
    except (KeyError, TypeError, ValueError):
        return None
    return v if v > 0 and math.isfinite(v) else None


def save_cached_sec_per_unit(path: Path, task_type: str, sec_per_unit: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
    data[task_type] = {
        "sec_per_unit": float(sec_per_unit),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")


def default_eta_cache_path() -> Path:
    from fish.config import CONFIG_DIR

    return CONFIG_DIR / "eta_unit_rates.json"
