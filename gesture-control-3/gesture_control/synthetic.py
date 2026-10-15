"""A parametric, *rotation-aware* synthetic hand.

Purpose
-------
This module is not a toy. It is the test fixture that makes the recognition
logic verifiable:

* Unit tests drive every gesture through the real pose/classification pipeline
  at many rotations and scales, which is how the original "compare tip.y to
  pip.y" logic is *proven* to be inadequate and the replacement proven correct.
* ``--self-test`` runs those same cases on the user's machine.
* A synthetic detector built on this module lets the whole application --
  smoothing, debouncing, action dispatch, overlay -- run end to end with no
  camera, which is how the integration tests and the HUD screenshot are made.

The hand is built in its own local frame (origin at the wrist, fingers pointing
"up"), curled by steering each phalanx toward the palm centre, and only then
rotated, scaled and translated into normalised image coordinates. Because the
curl is applied in the local frame, a hand rotated 90 degrees is *geometrically
identical* to an unrotated one -- so any recognition failure under rotation is a
genuine bug in the recognition code, not an artefact of the fixture.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .types import FINGER_JOINTS, FINGERS, NUM_LANDMARKS, Finger, Handedness

__all__ = [
    "FingerModel",
    "HAND_MODEL",
    "make_hand",
    "ScriptedKeyframe",
    "GestureScript",
    "DEFAULT_SCRIPT",
    "extended_curls",
    "curled_curls",
]

# --------------------------------------------------------------------------- #
# Model definition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FingerModel:
    """Geometry of one finger in the hand's local frame (palm lengths, y down).

    Attributes:
        root: position of the metacarpal head (the knuckle).
        direction: unit direction of a fully extended finger.
        lengths: proximal, middle and distal phalanx lengths.
        curl: maximum rotation applied at each of the three joints, in degrees,
            when ``curl=1``. The rotation is always *toward the palm centre*,
            which is what lets one code path handle the thumb's very different
            range of motion.
        flexes: whether the finger participates in fist-curl at all. The thumb's
            fold is mostly abduction-driven, so it steers less.
    """

    root: tuple[float, float]
    direction: tuple[float, float]
    lengths: tuple[float, float, float]
    curl: tuple[float, float, float]
    abducted_direction: tuple[float, float] | None = None


def _unit(x: float, y: float) -> tuple[float, float]:
    norm = math.hypot(x, y)
    return (x / norm, y / norm)


#: Anatomically plausible proportions. Distances are in units of "palm length"
#: (wrist to middle knuckle), measured off the MediaPipe landmark layout.
HAND_MODEL: Mapping[Finger, FingerModel] = {
    Finger.THUMB: FingerModel(
        root=(-0.28, -0.10),
        direction=_unit(-0.26, -0.97),          # adducted: tucked along the index
        abducted_direction=_unit(-0.82, -0.57),  # abducted: held out from the palm
        lengths=(0.30, 0.26, 0.20),
        curl=(50.0, 62.0, 52.0),
    ),
    Finger.INDEX: FingerModel(
        root=(-0.30, -0.88),
        direction=_unit(-0.10, -0.995),
        lengths=(0.40, 0.25, 0.18),
        curl=(72.0, 88.0, 66.0),
    ),
    Finger.MIDDLE: FingerModel(
        root=(0.00, -0.96),
        direction=(0.0, -1.0),
        lengths=(0.44, 0.27, 0.19),
        curl=(72.0, 88.0, 66.0),
    ),
    Finger.RING: FingerModel(
        root=(0.28, -0.92),
        direction=_unit(0.10, -0.995),
        lengths=(0.41, 0.25, 0.18),
        curl=(72.0, 88.0, 66.0),
    ),
    Finger.PINKY: FingerModel(
        root=(0.52, -0.76),
        direction=_unit(0.22, -0.975),
        lengths=(0.31, 0.19, 0.14),
        curl=(72.0, 88.0, 66.0),
    ),
}

#: Where curled fingers aim. Roughly the middle of the palm, nudged toward the
#: wrist so fingertips fold *into* the hand rather than through it.
PALM_TARGET = (0.06, -0.60)


def extended_curls() -> dict[Finger, float]:
    """Curl values for a flat, open hand."""
    return dict.fromkeys(FINGERS, 0.0)


def curled_curls() -> dict[Finger, float]:
    """Curl values for a closed fist."""
    return dict.fromkeys(FINGERS, 1.0)


# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #


def _rotate(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cos, sin = math.cos(radians), math.sin(radians)
    return np.array(
        [vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos],
        dtype=np.float64,
    )


def _base_direction(model: FingerModel, abduct: float) -> np.ndarray:
    """Starting direction of the proximal phalanx."""
    if model.abducted_direction is None:
        return np.array(model.direction, dtype=np.float64)
    base = np.array(model.direction, dtype=np.float64)
    out = np.array(model.abducted_direction, dtype=np.float64)
    blended = base * (1.0 - abduct) + out * abduct
    return blended / max(float(np.linalg.norm(blended)), 1e-9)


def _chain(model: FingerModel, direction: np.ndarray, curl: float, sign: float) -> np.ndarray:
    """Fold one finger by rotating each joint ``sign * curl * model.curl[k]``."""
    position = np.array(model.root, dtype=np.float64)
    heading = np.asarray(direction, dtype=np.float64).copy()
    joints = [position.copy()]
    for index in range(3):
        if curl > 0.0:
            heading = _rotate(heading, sign * curl * model.curl[index])
        position = position + heading * model.lengths[index]
        joints.append(position.copy())
    return np.asarray(joints, dtype=np.float64)


def _fold_sign(model: FingerModel, direction: np.ndarray, curl: float) -> float:
    """Pick the fold direction that actually brings the fingertip into the palm.

    Choosing by measurement instead of by a per-finger sign table is what makes
    the same code path fold the thumb *across* the palm and the pinky *inward*:
    a fixed sign that curls the index correctly curls the thumb out of the hand,
    which is exactly the kind of silent geometry error that makes a fixture lie
    about what the recogniser can do.
    """
    if curl <= 0.0:
        return 1.0
    target = np.array(PALM_TARGET, dtype=np.float64)
    distances = {
        sign: float(np.linalg.norm(_chain(model, direction, curl, sign)[-1] - target))
        for sign in (1.0, -1.0)
    }
    return 1.0 if distances[1.0] <= distances[-1.0] else -1.0


def _curl_finger(model: FingerModel, curl: float, abduct: float) -> np.ndarray:
    """Return the four joint positions of one finger in the local frame."""
    direction = _base_direction(model, abduct)
    return _chain(model, direction, curl, _fold_sign(model, direction, curl))


def make_hand(
    curls: Mapping[Finger, float] | None = None,
    *,
    thumb_out: float = 1.0,
    rotation_deg: float = 0.0,
    scale: float = 0.26,
    center: tuple[float, float] = (0.5, 0.86),
    aspect_ratio: float = 16 / 9,
    mirror: bool = False,
) -> np.ndarray:
    """Build a ``(21, 3)`` array of normalised landmarks for a synthetic hand.

    Args:
        curls: per-finger curl in ``[0, 1]``; ``0`` extended, ``1`` fully closed.
            Missing fingers default to extended.
        thumb_out: ``0`` tucks the thumb against the index, ``1`` holds it out.
        rotation_deg: in-plane rotation about the wrist. Recognition must be
            invariant to this, and the tests assert exactly that.
        scale: palm length as a fraction of the frame height.
        center: normalised position of the wrist in the frame.
        aspect_ratio: frame width / height, so the hand is not stretched.
        mirror: flip left/right, i.e. simulate the other hand.

    Returns:
        MediaPipe-shaped normalised landmarks (``x`` by width, ``y`` by height).
    """
    curls = dict(curls or {})
    abduct = float(np.clip(thumb_out, 0.0, 1.0))

    local = np.zeros((NUM_LANDMARKS, 2), dtype=np.float64)
    for finger in FINGERS:
        model = HAND_MODEL[finger]
        amount = float(np.clip(curls.get(finger, 0.0), 0.0, 1.0))
        joints = _curl_finger(model, amount, abduct if finger is Finger.THUMB else 0.0)
        for landmark, position in zip(FINGER_JOINTS[finger], joints):
            local[landmark] = position

    # The wrist is the origin; fingers were modelled along -y (up the image).
    rotation = math.radians(rotation_deg)
    cos, sin = math.cos(rotation), math.sin(rotation)
    rotate = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    local = local @ rotate.T

    if mirror:
        local = local * np.array([-1.0, 1.0])

    normalised = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
    normalised[:, 0] = center[0] + (local[:, 0] * scale) / float(aspect_ratio)
    normalised[:, 1] = center[1] + (local[:, 1] * scale)
    return normalised


# --------------------------------------------------------------------------- #
# Scripted routine (used by the synthetic detector and the demo)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ScriptedHand:
    """One hand in a scripted pose."""

    curls: Mapping[Finger, float]
    thumb_out: float = 1.0
    handedness: Handedness = Handedness.RIGHT
    mirror: bool = False


@dataclass(frozen=True, slots=True)
class ScriptedKeyframe:
    """One pose in a scripted routine, optionally two-handed."""

    name: str
    duration: float
    curls: Mapping[Finger, float] = field(default_factory=extended_curls)
    thumb_out: float = 1.0
    hand_visible: bool = True
    handedness: Handedness = Handedness.RIGHT

    #: Optional second hand. ``None`` means the routine shows one hand here, so a
    #: script can move between one- and two-handed poses.
    second: Mapping[Finger, float] | None = None
    second_thumb_out: float = 1.0
    second_handedness: Handedness = Handedness.LEFT

    @property
    def two_handed(self) -> bool:
        return self.second is not None


def _pose(name: str, duration: float, up: Iterable[Finger], *, thumb_out: float = 1.0,
          visible: bool = True) -> ScriptedKeyframe:
    """Helper: every finger curled except those in ``up``."""
    up_set = set(up)
    curls = {finger: (0.0 if finger in up_set else 1.0) for finger in FINGERS}
    return ScriptedKeyframe(
        name=name, duration=duration, curls=curls, thumb_out=thumb_out, hand_visible=visible
    )


def _two_handed(name: str, duration: float, up: Iterable[Finger], *,
                thumb_out: float = 1.0) -> ScriptedKeyframe:
    """Helper: both hands making the same shape -- the symmetric form."""
    up_set = set(up)
    curls = {finger: (0.0 if finger in up_set else 1.0) for finger in FINGERS}
    return ScriptedKeyframe(
        name=name,
        duration=duration,
        curls=curls,
        thumb_out=thumb_out,
        second=dict(curls),
        second_thumb_out=thumb_out,
    )


#: A routine that visits every gesture the app knows about, with transitions,
#: plus a stretch with no hand at all so presence handling is exercised.
DEFAULT_SCRIPT: tuple[ScriptedKeyframe, ...] = (
    ScriptedKeyframe(name="absent", duration=0.9, hand_visible=False),
    _pose("open", 0.9, FINGERS),
    _pose("index", 0.9, [Finger.INDEX]),
    _pose("index+middle", 0.9, [Finger.INDEX, Finger.MIDDLE]),
    _pose("ring+pinky", 0.9, [Finger.RING, Finger.PINKY], thumb_out=0.0),
    _pose("pinky", 0.9, [Finger.PINKY], thumb_out=0.0),
    _pose("fist", 0.9, [], thumb_out=0.0),
    _pose("thumb up", 0.9, [Finger.THUMB], thumb_out=1.0),
    _pose("open", 0.6, FINGERS),
)

#: A two-handed routine that exercises the mode-change gestures.
#:
#: Both hands make the same shape, which is how the symmetric two-hand gestures
#: are defined. The quit gesture (both horns) is deliberately **not** included:
#: a routine that ends the session is not useful as a repeatable demo, and it is
#: covered by a dedicated test instead.
TWO_HAND_SCRIPT: tuple[ScriptedKeyframe, ...] = (
    ScriptedKeyframe(name="absent", duration=0.5, hand_visible=False),
    _pose("one hand", 0.7, [Finger.INDEX]),
    _two_handed("both open", 0.9, FINGERS),
    _pose("one hand", 0.6, [Finger.INDEX]),
    _two_handed("both peace", 0.9, [Finger.INDEX, Finger.MIDDLE], thumb_out=0.0),
    _pose("one hand", 0.6, [Finger.INDEX]),
    _two_handed("both shaka", 0.9, [Finger.THUMB, Finger.PINKY]),
    _pose("one hand", 0.6, [Finger.INDEX]),
    _two_handed("both fist", 0.9, [], thumb_out=0.0),
    _pose("one hand", 0.6, [Finger.INDEX]),
)

#: A routine that quits, for testing the two-hand quit gesture.
QUIT_SCRIPT: tuple[ScriptedKeyframe, ...] = (
    _pose("one hand", 0.4, [Finger.INDEX]),
    _two_handed("both horns", 2.0, [Finger.INDEX, Finger.PINKY], thumb_out=0.0),
)


class GestureScript:
    """A looping, smoothly interpolated sequence of hand poses.

    Interpolation matters: real hands do not teleport between gestures, so the
    debouncer must tolerate the ambiguous frames in between. Testing on
    teleporting keyframes would give the temporal layer an unfairly easy job.
    """

    def __init__(
        self,
        keyframes: Sequence[ScriptedKeyframe] = DEFAULT_SCRIPT,
        *,
        transition: float = 0.18,
        rotation_span: float = 30.0,
        rotation_period: float = 12.0,
        scale: float = 0.26,
        aspect_ratio: float = 16 / 9,
        mirror: bool = False,
    ) -> None:
        if not keyframes:
            raise ValueError("a script needs at least one keyframe")
        self.keyframes = tuple(keyframes)
        self.transition = max(0.0, transition)
        self.rotation_span = rotation_span
        self.rotation_period = max(1e-3, rotation_period)
        self.scale = scale
        self.aspect_ratio = aspect_ratio
        self.mirror = mirror
        self._starts = np.cumsum([0.0, *(k.duration for k in self.keyframes)])
        self.duration = float(self._starts[-1])

    # -- timeline ---------------------------------------------------------- #

    def _blend(self, elapsed: float) -> tuple[float, ScriptedKeyframe, ScriptedKeyframe]:
        """Return ``(alpha, from_frame, to_frame)`` for a time in the loop."""
        time = elapsed % self.duration
        index = int(np.searchsorted(self._starts, time, side="right")) - 1
        index = max(0, min(index, len(self.keyframes) - 1))
        current = self.keyframes[index]
        start = float(self._starts[index])
        if index + 1 < len(self.keyframes) and self.transition > 0.0:
            remaining = current.duration - (time - start)
            if remaining < self.transition:
                nxt = self.keyframes[index + 1]
                alpha = 1.0 - max(0.0, remaining) / self.transition
                return float(np.clip(alpha, 0.0, 1.0)), current, nxt
        return 0.0, current, current

    def state_at(self, elapsed: float) -> tuple[dict[Finger, float], float, bool, Handedness]:
        """Interpolated ``(curls, thumb_out, visible, handedness)`` of the first hand."""
        hands = self.hands_at(elapsed)
        if not hands:
            curls, _, _, _ = self._interpolate(elapsed)
            return curls, 0.0, False, Handedness.RIGHT
        first = hands[0]
        return dict(first.curls), first.thumb_out, True, first.handedness

    def _interpolate(
        self, elapsed: float
    ) -> tuple[dict[Finger, float], float, bool, Handedness]:
        """Blend the primary hand's shape, without deciding how many hands exist."""
        alpha, current, target = self._blend(elapsed)
        if current.hand_visible != target.hand_visible:
            # Never morph a hand in or out of existence; the frame where the hand
            # is genuinely absent is what the presence check must handle.
            visible = current.hand_visible if alpha < 0.5 else target.hand_visible
        else:
            visible = current.hand_visible

        curls: dict[Finger, float] = {}
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
        for finger in FINGERS:
            a = current.curls.get(finger, 0.0)
            b = target.curls.get(finger, a)
            curls[finger] = a + (b - a) * smooth

        thumb_out = current.thumb_out + (target.thumb_out - current.thumb_out) * alpha
        handedness = current.handedness if alpha < 0.5 else target.handedness
        return curls, float(thumb_out), visible, handedness

    def hands_at(self, elapsed: float) -> tuple[ScriptedHand, ...]:
        """Every hand shown at ``elapsed``: one, two, or none.

        The second hand exists only where the script gives it a shape. During a
        transition towards a two-handed pose it appears with the *current* shape
        rather than fading in, because a hand that is half-present is not a thing
        the detector can produce and would only test the fixture's imagination.
        """
        curls, thumb_out, visible, handedness = self._interpolate(elapsed)
        if not visible:
            return ()
        alpha, current, target = self._blend(elapsed)

        hands = [
            ScriptedHand(
                curls=curls,
                thumb_out=thumb_out,
                handedness=handedness,
                mirror=self.mirror,
            )
        ]

        if current.second is None:
            return tuple(hands)

        second_curls: dict[Finger, float] = {}
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        for finger in FINGERS:
            a = current.second.get(finger, 0.0)
            b = (target.second or current.second).get(finger, a)
            second_curls[finger] = a + (b - a) * smooth
        second_thumb = current.second_thumb_out
        if target.second is not None:
            second_thumb += (target.second_thumb_out - second_thumb) * alpha

        hands.append(
            ScriptedHand(
                curls=second_curls,
                thumb_out=float(second_thumb),
                handedness=current.second_handedness,
                # The second hand is mirrored, so the pair reads as a left and a
                # right hand rather than two copies of the same one.
                mirror=not self.mirror,
            )
        )
        return tuple(hands)

    @staticmethod
    def _centre_for(index: int) -> tuple[float, float]:
        """Where each hand sits, so two hands do not overlap."""
        return (0.5, 0.86) if index == 0 else (0.2, 0.88)

    def landmarks_at(self, elapsed: float) -> np.ndarray | None:
        """Normalised landmarks of the first hand, or ``None`` when none is shown."""
        hands = self.hands_at(elapsed)
        if not hands:
            return None
        return self.landmarks_for(hands[0], elapsed, 0)

    def all_landmarks_at(self, elapsed: float) -> tuple[np.ndarray, ...]:
        """Landmarks for every hand shown at ``elapsed``."""
        return tuple(
            self.landmarks_for(hand, elapsed, index)
            for index, hand in enumerate(self.hands_at(elapsed))
        )

    def landmarks_for(self, hand: ScriptedHand, elapsed: float, index: int) -> np.ndarray:
        """Landmarks for one scripted hand at ``index`` (0 = first)."""
        rotation = self.rotation_span * math.sin(2.0 * math.pi * elapsed / self.rotation_period)
        return make_hand(
            hand.curls,
            thumb_out=hand.thumb_out,
            rotation_deg=rotation,
            scale=self.scale,
            center=self._centre_for(index),
            aspect_ratio=self.aspect_ratio,
            mirror=hand.mirror,
        )

    def label_at(self, elapsed: float) -> str:
        _, current, target = self._blend(elapsed)
        return current.name if target is current else f"{current.name}->{target.name}"

    def timeline(self) -> list[tuple[float, float, str]]:
        """``(start, end, name)`` for every keyframe, for documentation."""
        return [
            (float(self._starts[i]), float(self._starts[i + 1]), frame.name)
            for i, frame in enumerate(self.keyframes)
        ]
