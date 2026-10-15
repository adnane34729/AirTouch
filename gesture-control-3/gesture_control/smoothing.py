"""Landmark smoothing: a One Euro filter per coordinate.

MediaPipe landmarks jitter by a pixel or two even when the hand is perfectly
still. The original script fed raw landmarks straight into threshold
comparisons, so near a threshold the finger states flickered and the UI text
strobed.

A plain exponential moving average fixes the flicker but adds lag, which makes
fast gestures feel mushy. The One Euro filter (Casiez, Roussel & Vogel, 2012)
avoids that trade-off: it runs a low cutoff when the signal is slow (heavy
smoothing, no jitter) and raises the cutoff in proportion to the estimated
velocity (light smoothing, low lag). It has three parameters, no training, and
is a few lines of arithmetic -- the right tool for a real-time landmark stream.

The filter is per-coordinate, so the 21x3 landmark array is filtered by 42
independent instances. Timestamps are used directly, which makes the response
independent of the camera's frame rate.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["OneEuroFilter", "LandmarkSmoother", "MultiHandSmoother"]

_TWO_PI = 2.0 * math.pi


class OneEuroFilter:
    """First-order low-pass filter with a velocity-adaptive cutoff.

    Args:
        min_cutoff: cutoff frequency (Hz) when the signal is still. Lower means
            smoother at rest but laggier. Must be > 0.
        beta: how aggressively velocity raises the cutoff. ``0`` degrades the
            filter to a plain EMA.
        d_cutoff: cutoff (Hz) for the internal derivative estimate.

    Example:
        >>> import math
        >>> f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        >>> _ = f(0.0, 0.0)
        >>> round(f(1.0, 0.1), 4) > 0
        True
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_prev", "_dx_prev", "_t_prev", "_seen")

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        if min_cutoff <= 0.0:
            raise ValueError("min_cutoff must be positive")
        if d_cutoff <= 0.0:
            raise ValueError("d_cutoff must be positive")
        if beta < 0.0:
            raise ValueError("beta must not be negative")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev = 0.0
        self._t_prev: float | None = None
        self._seen = 0

    @property
    def initialised(self) -> bool:
        return self._x_prev is not None

    @property
    def samples(self) -> int:
        return self._seen

    def reset(self) -> None:
        """Forget history, so the next sample is treated as a fresh start."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None
        self._seen = 0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (_TWO_PI * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, timestamp: float) -> float:
        """Filter ``value`` observed at ``timestamp`` (seconds, monotonic)."""
        value = float(value)
        timestamp = float(timestamp)

        if self._x_prev is None or self._t_prev is None:
            self._x_prev = value
            self._t_prev = timestamp
            self._seen = 1
            return value

        dt = timestamp - self._t_prev
        if not math.isfinite(dt) or dt <= 0.0:
            # Duplicate or out-of-order timestamps: hold the previous estimate
            # rather than dividing by zero. Also keeps the filter immune to a
            # clock that steps backwards.
            return self._x_prev

        # Estimate the derivative, itself low-pass filtered.
        raw_derivative = (value - self._x_prev) / dt
        derivative_alpha = self._alpha(self.d_cutoff, dt)
        derivative = derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * self._dx_prev

        # Faster motion -> higher cutoff -> less smoothing -> less lag.
        cutoff = self.min_cutoff + self.beta * abs(derivative)
        alpha = self._alpha(cutoff, dt)
        estimate = alpha * value + (1.0 - alpha) * self._x_prev

        self._x_prev = estimate
        self._dx_prev = derivative
        self._t_prev = timestamp
        self._seen += 1
        return estimate


class LandmarkSmoother:
    """Smooths a ``(21, 3)`` landmark array with one filter per coordinate.

    The ``z`` channel is passed through untouched: it is already the least
    reliable axis, is unused by the gesture logic, and filtering it would only
    cost time.
    """

    __slots__ = ("_filters", "_shape", "_enabled", "min_cutoff", "beta", "d_cutoff")

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_cutoff: float = 1.1,
        beta: float = 0.035,
        d_cutoff: float = 1.0,
    ) -> None:
        self._enabled = bool(enabled)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._filters: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        """Drop filter state; the next landmark set starts clean.

        Called whenever the hand leaves the frame. Without this, the next hand
        to appear would be dragged toward the position of the previous one.
        """
        self._filters = None
        self._shape = None

    def _ensure(self, shape: tuple[int, int]) -> np.ndarray:
        if self._filters is None or self._shape != shape:
            self._filters = np.empty(shape, dtype=object)
            for index in np.ndindex(shape):
                self._filters[index] = OneEuroFilter(self.min_cutoff, self.beta, self.d_cutoff)
            self._shape = shape
        return self._filters

    def smooth(self, landmarks: np.ndarray, timestamp: float) -> np.ndarray:
        """Return a smoothed copy of ``landmarks`` (shape preserved)."""
        array = np.asarray(landmarks, dtype=np.float64)
        if not self._enabled or array.ndim != 2:
            return array

        plane = array[:, :2]
        filters = self._ensure((plane.shape[0], 2))
        smoothed = np.array(array, dtype=np.float64, copy=True)
        for row in range(plane.shape[0]):
            for column in range(2):
                smoothed[row, column] = filters[row, column](
                    plane[row, column], timestamp
                )
        return smoothed


class MultiHandSmoother:
    """One :class:`LandmarkSmoother` per hand, keyed by a caller-supplied label.

    A single shared smoother cannot be used with two hands: its filters would be
    pulled between both hands' landmarks every frame, producing positions
    somewhere between them. Keying by hand keeps the histories separate, and lets
    a hand that leaves the frame have its history dropped without disturbing the
    other.

    The key must be **stable across frames** for a given hand -- usually its
    handedness, or a position-derived slot when the detector cannot decide. A key
    that changes every frame would reset the filter every frame, which is worse
    than no smoothing at all, so the caller owns that choice rather than this
    class guessing.
    """

    __slots__ = ("_smoothers", "_factory", "enabled")

    def __init__(
        self,
        *,
        enabled: bool = True,
        min_cutoff: float = 1.1,
        beta: float = 0.035,
        d_cutoff: float = 1.0,
    ) -> None:
        self.enabled = enabled
        self._factory = lambda: LandmarkSmoother(  # noqa: E731 - a tiny factory
            enabled=enabled, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
        )
        self._smoothers: dict[str, LandmarkSmoother] = {}

    def smooth(self, landmarks: np.ndarray, timestamp: float, *, key: str) -> np.ndarray:
        """Smooth one hand's landmarks, keeping history under ``key``."""
        smoother = self._smoothers.get(key)
        if smoother is None:
            smoother = self._factory()
            self._smoothers[key] = smoother
        return smoother.smooth(landmarks, timestamp)

    def forget(self, key: str) -> None:
        """Drop one hand's history (it left the frame)."""
        self._smoothers.pop(key, None)

    def reset(self) -> None:
        """Drop every hand's history."""
        self._smoothers.clear()

    def keys(self) -> tuple[str, ...]:
        return tuple(self._smoothers)

    @property
    def tracked_hands(self) -> int:
        return len(self._smoothers)
