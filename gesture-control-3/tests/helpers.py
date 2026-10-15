"""Shared test helpers."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from gesture_control.geometry import make_observation
from gesture_control.gestures import Requirement, build_gesture_set
from gesture_control.pose import PoseEstimator, estimate_pose
from gesture_control.synthetic import make_hand
from gesture_control.types import Finger, HandPose

#: Rotations every recognition test sweeps. Includes 90 and 180 degree cases,
#: which is precisely where a "tip above pip" test collapses.
ROTATIONS: tuple[float, ...] = (0.0, 15.0, 45.0, 90.0, 135.0, 180.0, -30.0, -90.0)

#: Palm lengths (fraction of frame height) to test scale invariance over.
SCALES: tuple[float, ...] = (0.12, 0.20, 0.26, 0.36, 0.48)

#: Aspect ratios of common cameras, plus an ultrawide.
ASPECTS: tuple[float, ...] = (4 / 3, 16 / 9, 21 / 9)


def pose_from_curls(
    curls: Mapping[Finger, float],
    *,
    thumb_out: float = 1.0,
    rotation: float = 0.0,
    scale: float = 0.26,
    aspect: float = 16 / 9,
    mirror: bool = False,
    estimator: PoseEstimator | None = None,
) -> HandPose:
    """Build a :class:`HandPose` from a curl specification."""
    raw = make_hand(
        curls,
        thumb_out=thumb_out,
        rotation_deg=rotation,
        scale=scale,
        aspect_ratio=aspect,
        mirror=mirror,
    )
    observation = make_observation(raw, aspect_ratio=aspect)
    if estimator is None:
        return estimate_pose(observation)
    return estimator.estimate(observation)


def extended(fingers: Iterable[Finger], thumb_out: float = 0.0) -> dict[Finger, float]:
    """Curl spec with exactly ``fingers`` extended."""
    wanted = set(fingers)
    all_fingers = (Finger.THUMB, Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
    return {finger: (0.0 if finger in wanted else 1.0) for finger in all_fingers}


def default_gesture_set():
    """The ``classic`` preset's only mode, as a gesture set."""
    from gesture_control.config import default_config

    return default_config("classic").build_gesture_set()


def mode_gesture_set(preset: str = "pro", mode: str | None = None):
    """Gesture set for one mode of a preset (the first mode by default)."""
    from gesture_control.config import default_config

    config = default_config(preset)
    chosen = next((m for m in config.modes if m.name == mode), config.modes[0])
    return chosen.build_gesture_set()


def pro_config():
    """The multi-mode preset, with no window and no real side effects."""
    from gesture_control.config import default_config

    config = default_config("pro")
    config.overlay.show_window = False
    config.actions.dry_run = True
    return config


def two_hand_set(preset: str = "pro"):
    from gesture_control.config import default_config

    return default_config(preset).build_two_hand_set()


def all_fingers() -> tuple[Finger, ...]:
    return _ALL_FINGERS


def fake_pose(up: Iterable[Finger]) -> HandPose:
    """A minimal :class:`HandPose` with a given set of fingers extended.

    Landmarks are all zero, so this is only usable for tests that care about
    finger *states* -- classification, tracking and action wiring. Anything that
    draws or measures geometry needs :func:`real_pose`.
    """
    from gesture_control.types import FingerReading, HandObservation

    wanted = set(up)
    observation = HandObservation(
        points=np.zeros((21, 2), dtype=np.float64),
        raw=np.zeros((21, 3), dtype=np.float64),
    )
    readings = {
        finger: FingerReading(
            finger=finger,
            curl=178.0 if finger in wanted else 62.0,
            reach=1.6 if finger in wanted else 0.8,
            extended=finger in wanted,
            abduction=0.8 if finger in wanted else 0.2,
        )
        for finger in (Finger.THUMB, Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)
    }
    return HandPose(observation=observation, readings=readings)


def real_pose(
    up: Iterable[Finger] | None = None,
    *,
    thumb_out: float = 0.0,
    rotation: float = 0.0,
    scale: float = 0.26,
    aspect: float = 16 / 9,
) -> HandPose:
    """A pose with genuine landmark geometry, for drawing and measurement tests."""
    from gesture_control.types import Handedness

    wanted = set(up or ())
    curls = {f: (0.0 if f in wanted else 1.0) for f in _ALL_FINGERS}
    raw = make_hand(
        curls,
        thumb_out=thumb_out,
        rotation_deg=rotation,
        scale=scale,
        aspect_ratio=aspect,
    )
    observation = make_observation(
        raw, aspect_ratio=aspect, handedness=Handedness.RIGHT, handedness_score=1.0
    )
    return estimate_pose(observation)


_ALL_FINGERS = (
    Finger.THUMB,
    Finger.INDEX,
    Finger.MIDDLE,
    Finger.RING,
    Finger.PINKY,
)


__all__ = [
    "ROTATIONS",
    "SCALES",
    "ASPECTS",
    "pose_from_curls",
    "extended",
    "default_gesture_set",
    "mode_gesture_set",
    "pro_config",
    "two_hand_set",
    "fake_pose",
    "real_pose",
    "all_fingers",
    "Requirement",
    "build_gesture_set",
    "make_hand",
    "np",
]
