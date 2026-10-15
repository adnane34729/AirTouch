"""Turning landmarks into a hand pose: the five up/down finger decisions.

The decision rules here were calibrated against the synthetic hand model (see
``tests/test_geometry.py`` and ``--self-test``), and every threshold is exposed
in the config file with the measured margins documented, so a user whose hands
differ from the model can retune from the live HUD readout instead of guessing.

Two ideas carry the accuracy:

**Hysteresis.** Each finger has separate "definitely up" and "definitely down"
thresholds. Anything between them keeps the previous answer. Without this, a
finger hovering near a threshold flickers up/down every frame and machine-guns
actions; the original had no hysteresis at all.

**Two independent metrics.** ``curl`` (the angle at the finger's own knuckle-ward
joint) and ``reach`` (how far the tip projects along the palm axis). Curl alone
cannot see a finger that is straight but folded down at the knuckle -- a very
common way to close a fist -- because the finger's own joints really are
straight. Reach catches that case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import geometry
from .types import (
    FINGERS,
    Finger,
    FingerReading,
    HandObservation,
    HandPose,
    Handedness,
)

__all__ = [
    "FingerThresholds",
    "ThumbThresholds",
    "Thresholds",
    "PoseEstimator",
    "estimate_pose",
]


@dataclass(frozen=True, slots=True)
class FingerThresholds:
    """Hysteresis bands for the four non-thumb fingers.

    Calibrated margins against the hand model, in ``(extended, curled)`` form::

        curl   180.0   64.6 - 65.4     band [120, 150] sits in a >45 deg gap
        reach  1.47 - 1.97   0.68 - 0.77   band [0.95, 1.15] sits in a >0.3 gap
    """

    curl_up: float = 150.0
    """Minimum joint angle for a definite "up" (~180 straight)."""

    curl_down: float = 120.0
    """Maximum joint angle for a definite "down"."""

    reach_up: float = 1.15
    """Minimum palm-axis reach for a definite "up", in palm lengths."""

    reach_down: float = 0.95
    """Maximum palm-axis reach for a definite "down"."""

    @property
    def curl_mid(self) -> float:
        return 0.5 * (self.curl_up + self.curl_down)

    @property
    def reach_mid(self) -> float:
        return 0.5 * (self.reach_up + self.reach_down)


@dataclass(frozen=True, slots=True)
class ThumbThresholds:
    """Hysteresis bands for the thumb.

    The thumb needs ``curl`` *and* ``abduction`` because neither alone works.
    Measured on the model, in ``(thumb_curl, abduction)`` form::

        straight + held out    (thumbs up)  180.0, 0.74
        straight + tucked      (by the index) 180.0, 0.19
        folded across the palm                118.0, 0.98

    Abduction alone ranks the *folded* thumb as more "out" than the thumbs-up,
    so it can only be trusted when combined with curl.
    """

    curl_up: float = 150.0
    curl_down: float = 130.0
    abduction_up: float = 0.35
    abduction_down: float = 0.28

    @property
    def curl_mid(self) -> float:
        return 0.5 * (self.curl_up + self.curl_down)

    @property
    def abduction_mid(self) -> float:
        return 0.5 * (self.abduction_up + self.abduction_down)


@dataclass(frozen=True, slots=True)
class Thresholds:
    """All pose-decision thresholds."""

    finger: FingerThresholds = field(default_factory=FingerThresholds)
    thumb: ThumbThresholds = field(default_factory=ThumbThresholds)


DEFAULT_THRESHOLDS = Thresholds()


def _decide(
    up: bool,
    down: bool,
    previous: bool | None,
    fallback: bool,
) -> bool:
    """Apply hysteresis to a two-metric decision.

    Args:
        up: the finger definitely meets the "extended" criteria.
        down: the finger definitely meets the "curled" criteria.
        previous: last frame's answer, or ``None`` when a hand was just acquired.
        fallback: answer to use the first time a hand is seen.

    Returns:
        The stable answer. ``down`` is checked first so that a finger which
        somehow satisfies both bands resolves conservatively (never claim a
        finger is up when it might be down, since "up" triggers real actions).
    """
    if down:
        return False
    if up:
        return True
    if previous is not None:
        return previous
    return fallback


@dataclass
class _FingerMemory:
    states: dict[Finger, bool] = field(default_factory=dict)


class PoseEstimator:
    """Stateful landmark -> :class:`HandPose` converter.

    Stateful because hysteresis needs the previous answer. Memory is kept per
    handedness so that two hands in frame do not share (and therefore corrupt)
    each other's hysteresis. :meth:`reset` clears it whenever hands leave the
    frame, so a newly acquired hand is never judged against a stale pose.
    """

    def __init__(self, thresholds: Thresholds | None = None) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self._memory: dict[Handedness, _FingerMemory] = {}
        self.reset_count = 0

    # -- lifecycle --------------------------------------------------------- #

    def reset(self, handedness: Handedness | None = None) -> None:
        """Forget previous states, for one hand or all of them."""
        if handedness is None:
            if self._memory:
                self.reset_count += 1
            self._memory.clear()
        else:
            self._memory.pop(handedness, None)
        return None

    def _memory_for(self, handedness: Handedness) -> _FingerMemory:
        memory = self._memory.get(handedness)
        if memory is None:
            memory = _FingerMemory()
            self._memory[handedness] = memory
        return memory

    # -- estimation -------------------------------------------------------- #

    def estimate(self, observation: HandObservation) -> HandPose:
        """Classify every finger of one hand, updating hysteresis state."""
        points = observation.points
        memory = self._memory_for(observation.handedness)
        finger_thresholds = self.thresholds.finger
        thumb_thresholds = self.thresholds.thumb

        # Hand-relative quantities are computed once and shared by all fingers.
        scale = geometry.palm_size(points)
        axis = geometry.palm_axis(points)

        readings: dict[Finger, FingerReading] = {}
        for finger in FINGERS:
            curl = geometry.finger_curl(points, finger)
            reach = geometry.finger_reach(points, finger, axis=axis, scale=scale)
            previous = memory.states.get(finger)

            if finger is Finger.THUMB:
                abduction = geometry.thumb_abduction(points, scale=scale)
                extended = _decide(
                    up=curl >= thumb_thresholds.curl_up
                    and abduction >= thumb_thresholds.abduction_up,
                    down=curl <= thumb_thresholds.curl_down
                    or abduction <= thumb_thresholds.abduction_down,
                    previous=previous,
                    fallback=curl >= thumb_thresholds.curl_mid
                    and abduction >= thumb_thresholds.abduction_mid,
                )
            else:
                abduction = math.nan
                extended = _decide(
                    up=curl >= finger_thresholds.curl_up and reach >= finger_thresholds.reach_up,
                    down=curl <= finger_thresholds.curl_down
                    or reach <= finger_thresholds.reach_down,
                    previous=previous,
                    fallback=curl >= finger_thresholds.curl_mid
                    and reach >= finger_thresholds.reach_mid,
                )

            memory.states[finger] = extended
            readings[finger] = FingerReading(
                finger=finger,
                curl=curl,
                reach=reach,
                extended=extended,
                abduction=abduction,
            )

        return HandPose(observation=observation, readings=readings)


def estimate_pose(
    observation: HandObservation,
    thresholds: Thresholds | None = None,
) -> HandPose:
    """One-shot, stateless estimation (no hysteresis).

    Convenient for tests and diagnostics, where a fresh decision from the
    mid-band fallback is wanted rather than a remembered one.
    """
    return PoseEstimator(thresholds).estimate(observation)
