"""Hand-shape pictograms for the HUD legend.

A legend that reads ``v ^ v v v`` requires you to know which column is which
finger, what ``^`` and ``v`` mean, and to translate that into a hand position in
your head. That is a notation, not documentation.

These icons are drawn from the *same synthetic hand model the tests use*, so a
legend icon is geometrically the shape the recogniser is looking for rather than
a separate drawing that could drift away from it. Each icon is computed once and
cached, because a legend redraws every frame and the shape never changes.

A finger the gesture ignores (``any``) is drawn half-curled in the neutral
palette colour, so "this finger does not matter" is visible instead of implied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .gestures import Requirement
from .synthetic import make_hand
from .types import FINGER_JOINTS, FINGERS, Finger

__all__ = ["ShapeIcon", "IconLibrary"]


@dataclass(frozen=True, slots=True)
class ShapeIcon:
    """A hand shape reduced to drawable, position-independent geometry.

    Attributes:
        points: 21 ``(x, y)`` pairs normalised into ``[0, 1]`` on each axis, so
            the caller can map them into any icon box without distortion as long
            as the box uses :attr:`aspect`.
        aspect: width / height of the shape's bounding box. The caller derives
            the drawing width from the height it wants.
        extended: which fingers are raised, used to colour the bones.
        unspecified: which fingers the gesture does not constrain.
    """

    points: tuple[tuple[float, float], ...]
    aspect: float
    extended: tuple[bool, ...]
    unspecified: tuple[bool, ...]

    def finger_bones(self) -> tuple[tuple[Finger, tuple[tuple[int, int], ...]], ...]:
        """Bones per finger, so each can be drawn in its own colour."""
        return tuple(
            (
                finger,
                tuple(
                    (joints[index], joints[index + 1]) for index in range(len(joints) - 1)
                ),
            )
            for finger, joints in ((f, FINGER_JOINTS[f]) for f in FINGERS)
        )


class IconLibrary:
    """Builds and caches :class:`ShapeIcon` objects."""

    def __init__(self, cache_size: int = 256) -> None:
        self._cache: dict[tuple, ShapeIcon] = {}
        self._cache_size = cache_size

    def get(self, requirements: Mapping[Finger, Requirement]) -> ShapeIcon:
        """Icon for a requirement table, computed once and reused."""
        key = tuple(requirements.get(finger, Requirement.ANY).value for finger in FINGERS)
        icon = self._cache.get(key)
        if icon is None:
            icon = self._build(requirements)
            if len(self._cache) >= self._cache_size:
                self._cache.clear()
            self._cache[key] = icon
        return icon

    @staticmethod
    def _build(requirements: Mapping[Finger, Requirement]) -> ShapeIcon:
        curls: dict[Finger, float] = {}
        extended: list[bool] = []
        unspecified: list[bool] = []
        for finger in FINGERS:
            requirement = requirements.get(finger, Requirement.ANY)
            if requirement is Requirement.UP:
                curls[finger] = 0.0
                extended.append(True)
                unspecified.append(False)
            elif requirement is Requirement.DOWN:
                curls[finger] = 1.0
                extended.append(False)
                unspecified.append(False)
            else:
                # Halfway, and drawn in the neutral colour: neither up nor down,
                # because the gesture does not care.
                curls[finger] = 0.5
                extended.append(False)
                unspecified.append(True)

        thumb_requirement = requirements.get(Finger.THUMB, Requirement.ANY)
        if thumb_requirement is Requirement.UP:
            thumb_out = 1.0
        elif thumb_requirement is Requirement.DOWN:
            thumb_out = 0.0
        else:
            thumb_out = 0.5

        # scale=1, centre at the origin and aspect 1 mean the model returns its
        # own local coordinates, which is exactly what we want to normalise.
        local = make_hand(
            curls,
            thumb_out=thumb_out,
            rotation_deg=0.0,
            scale=1.0,
            center=(0.0, 0.0),
            aspect_ratio=1.0,
        )

        xs = local[:, 0]
        ys = local[:, 1]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())
        width = max(max_x - min_x, 1e-6)
        height = max(max_y - min_y, 1e-6)

        points = tuple(
            ((float(x) - min_x) / width, (float(y) - min_y) / height)
            for x, y in zip(xs, ys)
        )
        return ShapeIcon(
            points=points,
            aspect=width / height,
            extended=tuple(extended),
            unspecified=tuple(unspecified),
        )
