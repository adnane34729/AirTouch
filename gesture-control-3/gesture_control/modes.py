"""Modes and two-hand gestures.

**Modes** are the answer to "the same hand shape should do different things".
A :class:`Mode` owns its own gesture table, so ``index`` can mean *volume up* in
one mode and *next slide* in another. Exactly one mode is active at a time, and
the HUD always names it.

**Two-hand gestures** are the mode-control layer, deliberately separated from the
single-hand vocabulary:

* they need **both** hands, so they cannot be triggered by accident while you are
  gesturing normally with one hand;
* they are matched by :class:`TwoHandGestureSet` with its own, stricter ambiguity
  rules;
* the application gives them their own debouncer with a **longer arming time**,
  so switching modes or quitting requires a deliberate hold rather than a shape
  that happens to flash past.

Requiring both hands is the point: it makes the destructive actions (quit) and
the disorienting ones (mode changes) impossible to reach with one hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping, Sequence

from .gestures import (
    GestureError,
    GestureSet,
    GestureSpec,
    Requirement,
    requirements_match,
    requirements_symbols,
)
from .types import Finger, HandPose

__all__ = [
    "Mode",
    "ModeSet",
    "TwoHandGesture",
    "TwoHandGestureSet",
    "ModeError",
    "shape",
    "build_modes",
    "build_two_hand_gestures",
]


class ModeError(ValueError):
    """Raised when a mode or two-hand gesture definition is unusable."""


def shape(spec: str | Mapping[str, str] | Mapping[Finger, Requirement]) -> dict[Finger, Requirement]:
    """Alias for :func:`gesture_control.gestures.parse_requirements`.

    Present so mode tables read naturally: ``shape("index:up,pinky:up")``.
    """
    from .gestures import parse_requirements

    return parse_requirements(spec)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Mode:
    """One set of single-hand gestures, active together."""

    name: str
    label: str = ""
    description: str = ""
    gestures: tuple[GestureSpec, ...] = ()
    accent: str = "accent"
    """Key into the overlay palette, so each mode can be visually distinct."""

    def __post_init__(self) -> None:
        if not self.name:
            raise ModeError("a mode needs a name")
        if not self.label:
            object.__setattr__(self, "label", self.name.upper())

    @property
    def is_empty(self) -> bool:
        """True for a deliberately inert mode (a "rest your hand" mode)."""
        return not self.gestures

    def build_gesture_set(self) -> GestureSet | None:
        """Validated gesture set, or ``None`` for an empty mode.

        An empty mode is legal and useful -- it is how a "safe" mode that ignores
        single-hand gestures is expressed -- but :class:`GestureSet` refuses to be
        empty, because an empty set in a *config* is almost always a mistake.
        Returning ``None`` keeps that distinction explicit.
        """
        if not self.gestures:
            return None
        return GestureSet(specs=tuple(self.gestures))

    def classify(self, pose: HandPose | None) -> GestureSpec | None:
        gesture_set = self.build_gesture_set()
        return None if gesture_set is None else gesture_set.classify(pose)

    def guide_lines(self) -> list[str]:
        return [spec.guide_line() for spec in self.gestures]

    def __len__(self) -> int:
        return len(self.gestures)

    def __iter__(self) -> Iterator[GestureSpec]:
        return iter(self.gestures)


@dataclass
class ModeSet:
    """An ordered collection of modes, with switching semantics."""

    modes: tuple[Mode, ...] = ()
    wrap: bool = True
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.modes:
            raise ModeError("at least one mode is required")
        seen: set[str] = set()
        warnings: list[str] = []
        for mode in self.modes:
            if mode.name in seen:
                raise ModeError(f"duplicate mode name {mode.name!r}")
            seen.add(mode.name)
            # Building the set validates finger-requirement ambiguity inside the
            # mode, so a bad mode table fails at load rather than at run time.
            try:
                gesture_set = mode.build_gesture_set()
            except GestureError as exc:
                raise ModeError(f"mode {mode.name!r}: {exc}") from exc
            if gesture_set is not None:
                warnings.extend(f"[{mode.name}] {w}" for w in gesture_set.warnings)
        self.warnings = tuple(self.warnings) + tuple(warnings)

    # -- container behaviour ----------------------------------------------- #

    def __len__(self) -> int:
        return len(self.modes)

    def __iter__(self) -> Iterator[Mode]:
        return iter(self.modes)

    def __getitem__(self, index: int) -> Mode:
        return self.modes[index]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(mode.name for mode in self.modes)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(mode.label for mode in self.modes)

    def index_of(self, name: str) -> int:
        """Index of a mode by name, case-insensitively.

        Raises:
            ModeError: if there is no such mode.
        """
        wanted = name.strip().lower()
        for index, mode in enumerate(self.modes):
            if mode.name.lower() == wanted:
                return index
        raise ModeError(
            f"unknown mode {name!r}; available: {', '.join(self.names)}"
        )

    def step(self, current: int, delta: int) -> int:
        """Move ``delta`` modes from ``current``.

        Wraps when :attr:`wrap` is set (the default), otherwise clamps, so
        pressing "next mode" at the end either cycles or stops -- both are
        reasonable and the choice is the user's.
        """
        count = len(self.modes)
        if count == 0:  # pragma: no cover - __post_init__ forbids it
            return 0
        target = current + delta
        if self.wrap:
            return target % count
        return max(0, min(count - 1, target))

    @property
    def single_mode(self) -> bool:
        return len(self.modes) == 1


# --------------------------------------------------------------------------- #
# Two-hand gestures
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class TwoHandGesture:
    """A gesture that requires both hands.

    Two forms:

    **Symmetric** (the common case) -- ``shape`` must hold for *both* hands::

        TwoHandGesture("mode_next", "mode_next", shape=SHAPES["open"])

    Symmetric gestures ignore handedness entirely, which is deliberate: if the
    detector mislabels which hand is which, a symmetric gesture still matches.

    **Ordered** -- ``left`` and ``right`` describe each hand separately, for
    combinations such as "left open, right fist". Handedness mislabelling is
    still tolerated, because the matcher tries both assignments.

    Attributes are named to match :class:`~gesture_control.gestures.GestureSpec`
    so the debouncer can treat both kinds identically.
    """

    name: str
    action: str
    shape: Mapping[Finger, Requirement] | None = None
    left: Mapping[Finger, Requirement] | None = None
    right: Mapping[Finger, Requirement] | None = None
    label: str = ""
    description: str = ""
    repeatable: bool | None = False
    cooldown: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ModeError("a two-hand gesture needs a name")
        symmetric = self.shape is not None
        ordered = self.left is not None or self.right is not None
        if symmetric and ordered:
            raise ModeError(
                f"two-hand gesture {self.name!r} cannot be both symmetric (shape=) "
                f"and ordered (left=/right=)"
            )
        if not symmetric and not ordered:
            raise ModeError(
                f"two-hand gesture {self.name!r} needs either shape= for a symmetric "
                f"gesture or left=/right= for an ordered one"
            )
        if not self.label:
            object.__setattr__(self, "label", self.name.replace("_", " ").upper())
        if not self.constraints:
            raise ModeError(
                f"two-hand gesture {self.name!r} constrains no fingers and would match "
                f"any pair of hands"
            )

    # -- requirements ------------------------------------------------------ #

    @property
    def symmetric(self) -> bool:
        return self.shape is not None

    @property
    def constraints(self) -> int:
        """How many explicit finger requirements this gesture carries."""
        tables = [self.shape] if self.symmetric else [self.left, self.right]
        total = 0
        for table in tables:
            if not table:
                continue
            total += sum(1 for req in table.values() if req is not Requirement.ANY)
        return total

    def matches(self, first: HandPose, second: HandPose) -> bool:
        """True when the two observed hands satisfy this gesture.

        The arguments are in no particular order; the ordered form tries both
        assignments so a wrong handedness label cannot silently disable it.
        """
        if self.shape is not None:
            return requirements_match(self.shape, first) and requirements_match(
                self.shape, second
            )
        left = self.left or {}
        right = self.right or {}
        direct = requirements_match(left, first) and requirements_match(right, second)
        swapped = requirements_match(left, second) and requirements_match(right, first)
        return direct or swapped

    def hand_text(self) -> str:
        """Two-hand ASCII shape, e.g. ``"^ ^ ^ ^ ^  +  ^ ^ ^ ^ ^"``."""
        if self.shape is not None:
            one = requirements_symbols(self.shape)
            return f"{one}  +  {one}"
        left = requirements_symbols(self.left or {})
        right = requirements_symbols(self.right or {})
        return f"{left}  +  {right}"

    def guide_line(self) -> str:
        return f"{self.hand_text()}  {self.label}"

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name, "action": self.action}
        if self.shape is not None:
            payload["shape"] = {
                f.value: r.value for f, r in self.shape.items() if r is not Requirement.ANY
            }
        if self.left is not None:
            payload["left"] = {
                f.value: r.value for f, r in self.left.items() if r is not Requirement.ANY
            }
        if self.right is not None:
            payload["right"] = {
                f.value: r.value for f, r in self.right.items() if r is not Requirement.ANY
            }
        if self.label:
            payload["label"] = self.label
        if self.description:
            payload["description"] = self.description
        if self.repeatable is not None:
            payload["repeatable"] = bool(self.repeatable)
        if self.cooldown is not None:
            payload["cooldown"] = float(self.cooldown)
        return payload


def _compatible(
    first: Mapping[Finger, Requirement], second: Mapping[Finger, Requirement]
) -> bool:
    """True when two requirement tables do not contradict each other."""
    for finger, requirement in first.items():
        other = second.get(finger, Requirement.ANY)
        if requirement is Requirement.ANY or other is Requirement.ANY:
            continue
        if requirement is not other:
            return False
    return True


def _overlaps(first: TwoHandGesture, second: TwoHandGesture) -> bool:
    """True when some pair of hands could satisfy both gestures.

    A pair of hands satisfies a symmetric gesture when *both* hands fit its
    shape, and an ordered gesture when each hand fits its own side -- so the two
    gestures overlap exactly when the relevant tables are pairwise compatible.
    """
    if first.symmetric and second.symmetric:
        return _compatible(first.shape or {}, second.shape or {})
    if first.symmetric:
        shape = first.shape or {}
        return _compatible(shape, second.left or {}) and _compatible(shape, second.right or {})
    if second.symmetric:
        shape = second.shape or {}
        return _compatible(shape, first.left or {}) and _compatible(shape, first.right or {})
    direct = _compatible(first.left or {}, second.left or {}) and _compatible(
        first.right or {}, second.right or {}
    )
    swapped = _compatible(first.left or {}, second.right or {}) and _compatible(
        first.right or {}, second.left or {}
    )
    return direct or swapped


@dataclass
class TwoHandGestureSet:
    """Validated collection of two-hand gestures."""

    gestures: tuple[TwoHandGesture, ...] = ()
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for gesture in self.gestures:
            if gesture.name in seen:
                raise ModeError(f"duplicate two-hand gesture name {gesture.name!r}")
            seen.add(gesture.name)

        warnings: list[str] = []
        for index, first in enumerate(self.gestures):
            for second in self.gestures[index + 1:]:
                if not _overlaps(first, second):
                    continue
                if first.constraints == second.constraints:
                    raise ModeError(
                        f"two-hand gestures {first.name!r} and {second.name!r} can both "
                        f"match the same pair of hands and neither is more specific"
                    )
                winner, loser = (
                    (first, second)
                    if first.constraints > second.constraints
                    else (second, first)
                )
                warnings.append(
                    f"two-hand gesture {loser.name!r} overlaps {winner.name!r}; "
                    f"{winner.name!r} wins"
                )
        self.warnings = tuple(self.warnings) + tuple(warnings)

    def __len__(self) -> int:
        return len(self.gestures)

    def __iter__(self) -> Iterator[TwoHandGesture]:
        return iter(self.gestures)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(gesture.name for gesture in self.gestures)

    def by_name(self, name: str) -> TwoHandGesture | None:
        for gesture in self.gestures:
            if gesture.name == name:
                return gesture
        return None

    def classify(self, first: HandPose | None, second: HandPose | None) -> TwoHandGesture | None:
        """Best match for a pair of hands, or ``None``.

        Requires two hands by construction: a single-hand pose can never satisfy a
        two-hand gesture, which is what makes these gestures safe to bind to
        destructive actions.
        """
        if first is None or second is None:
            return None
        best: TwoHandGesture | None = None
        for gesture in self.gestures:
            if best is not None and gesture.constraints <= best.constraints:
                continue
            if gesture.matches(first, second):
                best = gesture
        return best

    def guide_lines(self) -> list[str]:
        return [gesture.guide_line() for gesture in self.gestures]


def build_modes(entries: Sequence[Mode | Mapping[str, object]]) -> tuple[Mode, ...]:
    """Build modes from already-typed objects or config mappings.

    Kept here rather than in :mod:`gesture_control.config` because modes are the
    unit of configuration: ``config`` calls this, not the other way round.
    """
    modes: list[Mode] = []
    for entry in entries:
        if isinstance(entry, Mode):
            modes.append(entry)
            continue
        gestures = entry.get("gestures") or ()
        from .config import build_gestures  # local import avoids a cycle

        modes.append(
            Mode(
                name=str(entry["name"]),
                label=str(entry.get("label") or ""),
                description=str(entry.get("description") or ""),
                gestures=build_gestures(gestures),  # type: ignore[arg-type]
                accent=str(entry.get("accent") or "accent"),
            )
        )
    return tuple(modes)


def build_two_hand_gestures(
    entries: Sequence[TwoHandGesture | Mapping[str, object]],
) -> tuple[TwoHandGesture, ...]:
    """Build two-hand gestures from typed objects or config mappings."""
    built: list[TwoHandGesture] = []
    for entry in entries:
        if isinstance(entry, TwoHandGesture):
            built.append(entry)
            continue
        try:
            name = str(entry["name"])
            action = str(entry["action"])
        except KeyError as exc:
            raise ModeError(
                f"two-hand gesture entry is missing the {exc.args[0]!r} key"
            ) from exc

        def _table(key: str) -> Mapping[Finger, Requirement] | None:
            raw = entry.get(key)
            if raw is None:
                return None
            try:
                return shape(raw)  # type: ignore[arg-type]
            except GestureError as exc:
                raise ModeError(f"two-hand gesture {name!r}: {exc}") from exc

        built.append(
            TwoHandGesture(
                name=name,
                action=action,
                shape=_table("shape"),
                left=_table("left"),
                right=_table("right"),
                label=str(entry.get("label") or ""),
                description=str(entry.get("description") or ""),
                repeatable=entry.get("repeatable"),  # type: ignore[arg-type]
                cooldown=entry.get("cooldown"),  # type: ignore[arg-type]
            )
        )
    return tuple(built)
