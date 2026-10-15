"""Rotation- and scale-invariant hand geometry.

Why this module exists
----------------------
The original script decided whether a finger was extended with::

    lm[tip].y < lm[pip].y - 0.02          # "points up therefore extended"

and decided the thumb with ``lm[4].x < lm[3].x``. Both are broken in ways that
are easy to miss on a good day and obvious on a bad one:

* **Not rotation invariant.** Tilt your hand 20 degrees, or gesture with the
  hand held sideways, and "tip above pip" stops being true even though the
  finger is perfectly straight.
* **Not scale invariant.** The ``- 0.02`` fudge is in normalised image units, so
  its effective size changes with how far the hand is from the camera.
* **Aspect-ratio naive.** Normalised ``x`` is divided by frame *width* and ``y``
  by frame *height*. Treating them as the same axis skews every angle by the
  camera aspect ratio (a 16:9 frame skews angles by ~1.8x horizontally).
* **Fails on the thumb.** Mirroring the frame means ``lm[4].x < lm[3].x``
  describes which way the hand faces, not whether the thumb is out.

Everything here works in an aspect-corrected metric space (see
:func:`to_metric`) and is expressed relative to the hand's own palm, so the
numbers do not move when the hand rotates, translates, or changes distance --
only when the fingers actually move.
"""

from __future__ import annotations

import numpy as np

from .types import (
    FINGER_JOINTS,
    HAND_CONNECTIONS,
    LONG_FINGERS,
    MIDDLE_MCP,
    NUM_LANDMARKS,
    PALM_MCP,
    THUMB_TIP,
    WRIST,
    Finger,
    HandObservation,
    Handedness,
)

__all__ = [
    "HAND_CONNECTIONS",
    "to_metric",
    "palm_size",
    "palm_axis",
    "palm_center",
    "joint_angle",
    "finger_curl",
    "finger_reach",
    "thumb_curl",
    "thumb_abduction",
    "finger_points",
    "hand_extent",
]

_EPS = 1e-9


def to_metric(raw: np.ndarray, aspect_ratio: float = 1.0) -> np.ndarray:
    """Convert normalised landmarks to aspect-corrected metric coordinates.

    Args:
        raw: ``(21, 3)`` or ``(21, 2)`` landmarks in ``[0, 1]`` normalised space.
        aspect_ratio: frame width / height.

    Returns:
        ``(N, 2)`` array where ``x`` is scaled by ``aspect_ratio``. Coordinates
        are in "frame heights": one unit equals the frame height.
    """
    points = np.asarray(raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < NUM_LANDMARKS:
        raise ValueError(f"expected at least ({NUM_LANDMARKS}, 2) landmarks, got {points.shape}")
    if not 0.05 <= aspect_ratio <= 20.0:
        raise ValueError(f"implausible aspect ratio {aspect_ratio!r}")

    metric = np.empty((points.shape[0], 2), dtype=np.float64)
    metric[:, 0] = points[:, 0] * aspect_ratio
    metric[:, 1] = points[:, 1]
    return metric


def palm_size(points: np.ndarray) -> float:
    """Mean knuckle-to-wrist span. The hand's own ruler.

    Averaging four independent spans rather than using a single one makes the
    scale estimate robust to one noisy knuckle, which matters because every
    threshold in this module is expressed in these units.
    """
    points = np.asarray(points, dtype=np.float64)
    wrist = points[WRIST]
    spans = [float(np.linalg.norm(points[PALM_MCP[f]] - wrist)) for f in LONG_FINGERS]
    return max(float(np.mean(spans)), _EPS)


def palm_axis(points: np.ndarray) -> np.ndarray:
    """Unit vector from the wrist to the middle knuckle.

    This is the hand's "forward" direction and the reference against which
    finger extension is measured, which is exactly what makes the result
    independent of how the hand is rotated in the image.
    """
    points = np.asarray(points, dtype=np.float64)
    axis = points[MIDDLE_MCP] - points[WRIST]
    norm = float(np.linalg.norm(axis))
    if norm < _EPS:
        return np.array([0.0, -1.0], dtype=np.float64)
    return axis / norm


def palm_center(points: np.ndarray) -> np.ndarray:
    """Centroid of the knuckles and the wrist: the hand's centre of mass."""
    points = np.asarray(points, dtype=np.float64)
    indices = [WRIST, *(PALM_MCP[f] for f in LONG_FINGERS)]
    return points[indices].mean(axis=0)


def joint_angle(points: np.ndarray, a: int, b: int, c: int) -> float:
    """Interior angle at ``b`` formed by ``a-b-c``, in degrees (0..180).

    Degenerate limbs (zero-length segments) return ``0.0`` rather than
    ``180.0``: a collapsed or occluded joint must never be mistaken for a
    straight one, because reading a finger as extended triggers real actions.
    """
    points = np.asarray(points, dtype=np.float64)
    u = points[a] - points[b]
    v = points[c] - points[b]
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < _EPS or nv < _EPS:
        return 0.0
    cosine = float(np.dot(u, v)) / (nu * nv)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cosine)))))


def finger_curl(points: np.ndarray, finger: Finger) -> float:
    """Curl angle of a finger in degrees; ~180 is straight, ~40 fully closed.

    Measured as ``angle(mcp, pip, tip)``. Referencing only the finger's own
    three landmarks -- never the wrist -- keeps this comparable across fingers
    with different fan-out angles, which is why a single pair of thresholds can
    serve all five.
    """
    mcp, pip, _dip, tip = FINGER_JOINTS[finger]
    return joint_angle(points, mcp, pip, tip)


def finger_reach(points: np.ndarray, finger: Finger, *, axis: np.ndarray | None = None,
                 scale: float | None = None) -> float:
    """How far the fingertip projects along the palm axis, in palm lengths.

    This is the second half of the extension decision. Curl alone cannot see a
    finger that is straight but folded down at the knuckle (a common way to make
    a fist), because the interphalangeal joints really are straight. Projecting
    the tip onto the palm axis catches exactly that case: a folded fingertip
    stops reaching past the knuckles.
    """
    points = np.asarray(points, dtype=np.float64)
    _mcp, _pip, _dip, tip = FINGER_JOINTS[finger]
    axis = palm_axis(points) if axis is None else axis
    scale = palm_size(points) if scale is None else scale
    return float(np.dot(points[tip] - points[WRIST], axis)) / scale


def thumb_curl(points: np.ndarray) -> float:
    """Straightness of the thumb: the worse of its two interphalangeal angles."""
    mcp, pip, dip, _tip = FINGER_JOINTS[Finger.THUMB]
    return min(joint_angle(points, mcp, pip, dip), joint_angle(points, pip, dip, _tip))


def thumb_abduction(points: np.ndarray, *, scale: float | None = None) -> float:
    """How far the thumb tip sits from the index knuckle, in palm lengths.

    Distinguishes a thumb held out from the palm (large) from one tucked flat
    against the side of the index finger (small) -- the case that a pure
    "is the thumb straight" test gets wrong.
    """
    points = np.asarray(points, dtype=np.float64)
    scale = palm_size(points) if scale is None else scale
    reference = points[PALM_MCP[Finger.INDEX]]
    return float(np.linalg.norm(points[THUMB_TIP] - reference)) / scale


def finger_points(points: np.ndarray, finger: Finger) -> np.ndarray:
    """The four ``(x, y)`` landmarks of one finger, for drawing."""
    return np.asarray(points, dtype=np.float64)[list(FINGER_JOINTS[finger])]


def hand_extent(points: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned bounding box ``(x0, y0, x1, y1)`` of the hand, in metric units."""
    points = np.asarray(points, dtype=np.float64)[:NUM_LANDMARKS]
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def make_observation(
    raw: np.ndarray,
    *,
    aspect_ratio: float = 1.0,
    handedness: Handedness = Handedness.UNKNOWN,
    handedness_score: float = 0.0,
    timestamp: float = 0.0,
) -> HandObservation:
    """Build a :class:`HandObservation`, converting to metric space once."""
    raw_array = np.asarray(raw, dtype=np.float64)
    return HandObservation(
        points=to_metric(raw_array, aspect_ratio),
        raw=raw_array,
        handedness=handedness,
        handedness_score=handedness_score,
        timestamp=timestamp,
        aspect_ratio=float(aspect_ratio),
    )
