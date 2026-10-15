"""Shared vocabulary: landmark topology, handedness, and immutable value types.

Landmark indices follow the MediaPipe hand model. Keeping the topology in one
place (instead of scattering magic numbers such as ``8`` and ``6`` through the
code as the original did) makes the geometry layer self-documenting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np

# --------------------------------------------------------------------------- #
# Landmark topology (MediaPipe hand model, 21 points)
# --------------------------------------------------------------------------- #

NUM_LANDMARKS = 21

WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

#: Bones of the palm slab, drawn in a neutral colour.
PALM_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 5),
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
)


class Handedness(str, Enum):
    """Which hand a detection belongs to.

    MediaPipe reports handedness assuming a *mirrored* (selfie) input, which is
    what a horizontally flipped webcam frame is, so the label is used as-is.
    """

    LEFT = "Left"
    RIGHT = "Right"
    UNKNOWN = "Unknown"

    @property
    def short(self) -> str:
        return {"Left": "L", "Right": "R", "Unknown": "?"}[self.value]


class Finger(str, Enum):
    """The five fingers, in anatomical order (thumb first)."""

    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"

    @property
    def display(self) -> str:
        return self.value.capitalize()

    @property
    def initial(self) -> str:
        return self.value[0].upper()


#: Canonical iteration order. Anything that reports "all fingers" uses this so
#: that the on-screen guide, the config dump, and the tests always agree.
FINGERS: tuple[Finger, ...] = (
    Finger.THUMB,
    Finger.INDEX,
    Finger.MIDDLE,
    Finger.RING,
    Finger.PINKY,
)

#: ``(mcp, pip, dip, tip)`` landmark indices per finger. For the thumb these are
#: ``(cmc, mcp, ip, tip)`` -- the thumb has one fewer phalanx, and using the same
#: four-slot shape lets the geometry code treat every finger uniformly.
FINGER_JOINTS: Mapping[Finger, tuple[int, int, int, int]] = {
    Finger.THUMB: (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    Finger.INDEX: (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    Finger.MIDDLE: (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    Finger.RING: (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    Finger.PINKY: (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

#: Knuckle landmarks, used to measure the size of the palm.
PALM_MCP: Mapping[Finger, int] = {finger: joints[0] for finger, joints in FINGER_JOINTS.items()}

#: The four fingers whose plane the palm axis is derived from. The thumb is
#: excluded on purpose: its metacarpal is not part of the palm slab.
LONG_FINGERS: tuple[Finger, ...] = (Finger.INDEX, Finger.MIDDLE, Finger.RING, Finger.PINKY)

#: Bones within each finger, drawn in that finger's own colour.
FINGER_CONNECTIONS: Mapping[Finger, tuple[tuple[int, int], ...]] = {
    finger: ((joints[0], joints[1]), (joints[1], joints[2]), (joints[2], joints[3]))
    for finger, joints in FINGER_JOINTS.items()
}

#: Every bone. MediaPipe ships an identical constant, but owning it keeps the
#: overlay independent of the detector backend -- the synthetic detector has no
#: MediaPipe object to borrow from.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = PALM_CONNECTIONS + tuple(
    bone for finger in FINGERS for bone in FINGER_CONNECTIONS[finger]
)


# --------------------------------------------------------------------------- #
# Value types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class HandObservation:
    """A single hand as produced by a detector, already in metric space.

    Attributes:
        points: ``(21, 2)`` array in *metric* space: ``x`` has been multiplied by
            the frame aspect ratio so that a unit step in ``x`` and ``y`` covers
            the same physical distance. Doing this once, at the boundary, is what
            lets the geometry layer compute meaningful angles -- the original
            compared raw normalised coordinates, which silently skews every angle
            by the camera aspect ratio.
        raw: ``(21, 3)`` original normalised landmarks, kept for drawing.
        aspect_ratio: frame width / height used to build ``points``. Carried so
            that anything working on ``raw`` (the smoother, for instance) can
            rebuild an identical observation without guessing.
        handedness: reported handedness of the hand.
        handedness_score: detector confidence for the handedness label.
        timestamp: capture time in seconds (monotonic clock).
    """

    points: np.ndarray
    raw: np.ndarray
    handedness: Handedness = Handedness.UNKNOWN
    handedness_score: float = 0.0
    timestamp: float = 0.0
    aspect_ratio: float = 1.0

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        if self.points.shape != (NUM_LANDMARKS, 2):
            raise ValueError(f"points must be ({NUM_LANDMARKS}, 2), got {self.points.shape}")

    @property
    def wrist(self) -> np.ndarray:
        return self.points[WRIST]


@dataclass(frozen=True, slots=True)
class FingerReading:
    """How extended one finger is, plus the raw numbers behind the decision.

    ``curl`` and ``reach`` are exposed rather than a single opaque boolean so the
    HUD can show *why* a finger was judged up or down, and so thresholds can be
    re-tuned from a live readout instead of by guesswork.
    """

    finger: Finger
    curl: float
    """Angle at the joint between the proximal and distal phalanx, in degrees.
    ~180 straight, well under 100 fully curled. Rotation invariant."""

    reach: float
    """Distance of the fingertip beyond the wrist, projected onto the palm axis
    and divided by palm size. ~0.8-1.1 curled, ~1.3+ extended."""

    extended: bool
    abduction: float = math.nan
    """Thumb only: fingertip distance to the index knuckle over palm size."""

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "finger": self.finger.value,
            "curl": round(self.curl, 2),
            "reach": round(self.reach, 3),
            "abduction": None if math.isnan(self.abduction) else round(self.abduction, 3),
            "extended": self.extended,
        }


@dataclass(frozen=True, slots=True)
class HandPose:
    """A hand observation interpreted as five up/down finger states."""

    observation: HandObservation
    readings: Mapping[Finger, FingerReading]

    def is_extended(self, finger: Finger) -> bool:
        reading = self.readings.get(finger)
        return bool(reading and reading.extended)

    @property
    def extended_fingers(self) -> frozenset[Finger]:
        return frozenset(f for f in FINGERS if self.is_extended(f))

    @property
    def signature(self) -> str:
        """Stable text key such as ``"index+middle"`` or ``"fist"``."""
        extended = [f for f in FINGERS if self.is_extended(f)]
        if not extended:
            return "fist"
        if len(extended) == len(FINGERS):
            return "open"
        return "+".join(f.value for f in extended)

    @property
    def handedness(self) -> Handedness:
        return self.observation.handedness
