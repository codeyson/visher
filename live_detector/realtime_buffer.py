"""
realtime_buffer.py
Thread-safe sliding-window buffer that accumulates raw PCM samples
and yields fixed-size chunks at a configurable hop interval.
"""

import threading
import numpy as np
from collections import deque
from typing import Iterator


class SlidingWindowBuffer:
    """
    Accumulates audio samples and emits overlapping windows.

    Parameters
    ----------
    window_samples : int
        Number of samples per inference window  (default 64 600 → ~2.7 s @ 24 kHz).
    hop_samples : int
        Step between successive windows          (default 24 000 → 1 s @ 24 kHz).
    dtype : np.dtype
        Sample dtype; internal storage is always float32.
    """

    def __init__(
        self,
        window_samples: int = 64_600,
        hop_samples: int   = 24_000,
        dtype                = np.float32,
    ):
        self.window_samples = window_samples
        self.hop_samples    = hop_samples
        self.dtype          = dtype

        self._buf: deque[float] = deque(maxlen=None)   # unbounded raw queue
        self._lock = threading.Lock()
        self._samples_since_last_hop = 0

    # ── public API ────────────────────────────────────────────────────────

    def push(self, samples: np.ndarray) -> None:
        """Add new samples (any shape / dtype – will be flattened & cast)."""
        samples = np.asarray(samples, dtype=np.float32).flatten()
        with self._lock:
            self._buf.extend(samples.tolist())
            self._samples_since_last_hop += len(samples)

    def ready(self) -> bool:
        """True when the buffer contains at least one full window."""
        with self._lock:
            return (
                len(self._buf) >= self.window_samples
                and self._samples_since_last_hop >= self.hop_samples
            )

    def get_window(self) -> np.ndarray | None:
        """
        If a full window is available, return it as a (window_samples,) float32
        array and advance the internal pointer by hop_samples.
        Returns None if not enough data yet.
        """
        with self._lock:
            if (
                len(self._buf) < self.window_samples
                or self._samples_since_last_hop < self.hop_samples
            ):
                return None

            chunk = np.array(list(self._buf)[:self.window_samples], dtype=np.float32)

            # Advance: drop hop_samples from front
            for _ in range(self.hop_samples):
                if self._buf:
                    self._buf.popleft()
            self._samples_since_last_hop = 0

            return chunk

    def drain_windows(self) -> Iterator[np.ndarray]:
        """Yield all available complete windows (non-blocking)."""
        while self.ready():
            w = self.get_window()
            if w is not None:
                yield w

    def reset(self) -> None:
        """Flush all buffered samples."""
        with self._lock:
            self._buf.clear()
            self._samples_since_last_hop = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class TemporalSmoother:
    """
    Reduces prediction jitter by majority-voting or averaging
    the last N binary predictions.

    Parameters
    ----------
    window_size : int  –  number of recent predictions to consider.
    mode        : str  –  'majority' or 'mean'.
    """

    def __init__(self, window_size: int = 5, mode: str = "majority"):
        assert mode in ("majority", "mean"), "mode must be 'majority' or 'mean'"
        self.window_size = window_size
        self.mode        = mode
        self._preds: deque[int]   = deque(maxlen=window_size)
        self._confs : deque[float] = deque(maxlen=window_size)

    def update(self, label: int, confidence: float) -> tuple[int, float]:
        """
        Add a new prediction; return the smoothed (label, confidence).
        """
        self._preds.append(label)
        self._confs.append(confidence)

        if self.mode == "majority":
            smoothed_label = int(round(np.mean(self._preds)))
        else:
            smoothed_label = int(np.mean(self._preds) >= 0.5)

        smoothed_conf = float(np.mean(self._confs))
        return smoothed_label, smoothed_conf

    def reset(self) -> None:
        self._preds.clear()
        self._confs.clear()