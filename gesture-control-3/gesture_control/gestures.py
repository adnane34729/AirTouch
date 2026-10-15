"""Gesture definitions and classification.

Two structural fixes over the original script live here.

**Tri-state requirements.** The original's ``check_fingers(lm, up_tips)`` was
all-or-nothing: it demanded that every finger not listed be *down*, including
the thumb. So "index finger up" silently required "thumb exactly down" too, and
the gesture failed whenever the thumb drifted out slightly -- which is most of
the time. Here each finger is required to be UP, DOWN, or **ANY** (ignored), so
a gesture constrains only the fingers it actually cares about.

**Declarative specs, generated guide.** The on-screen gesture list used to be a
hand-written string literal that duplicated the recogniser and could drift out of
sync with it. Now the guide is rendered from the same specs the recogniser uses,
so it cannot lie. Definitions also validate at load time: two specs that could
both match the same pose are rejected (or resolved by specificity and reported),
instead of silently shadowing one another as the original's nested ``if``
chain did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Iterator, Mapping, Sequence

from .types import FINGERS, Finger, HandPose

__all__ = [
    "Requirement",
    "GestureSpec",
    "GestureSet",
    "GestureError",
    "parse_requirements",
    "requirements_match",
    "requirements_symbols",
    "requirements_text",
]


class GestureError(ValueError):
    """Raised when a gesture set is ambiguous or malformed."""


class Requirement(str, Enum):
    """What a gesture demands of one finger."""

    UP = "up"
    DOWN = "down"
    ANY = "any"

    @property
    def symbol(self) -> str:
        return {"up": "^", "down": "v", "any": "-"}[self.value]


def requirements_match(requirements: Mapping[Finger, Requirement], pose: HandPose) -> bool:
    """True when ``pose`` satisfies every non-ANY entry of ``requirements``.

    Shared by single-hand gestures and each hand of a two-hand gesture, so both
    interpret a requirement table in exactly the same way.
    """
    for finger, requirement in requirements.items():
        if requirement is Requirement.ANY:
            continue
        if pose.is_extended(finger) != (requirement is Requirement.UP):
            return False
    return True


def requirements_symbols(requirements: Mapping[Finger, Requirement]) -> str:
    """Compact ``"v ^ v v v"`` rendering, thumb first, for the HUD."""
    return " ".join(
        requirements.get(finger, Requirement.ANY).symbol for finger in FINGERS
    )


def requirements_text(requirements: Mapping[Finger, Requirement]) -> str:
    """Human-readable form, e.g. ``"index: up, middle: down"``."""
    parts = [
        f"{finger.value}: {requirement.value}"
        for finger, requirement in requirements.items()
        if requirement is not Requirement.ANY
    ]
    return ", ".join(parts) if parts else "anything"


def parse_requirements(spec: str | Mapping[str, str] | Mapping[Finger, Requirement]) -> dict[Finger, Requirement]:
    """Parse several accepted shorthand forms into a requirement mapping.

    Accepted::

        "index:up,middle:down"            # compact config string
        {"index": "up", "middle": "down"}  # TOML table
        {Finger.INDEX: Requirement.UP}     # already parsed

    Fingers that are not mentioned default to :attr:`Requirement.ANY`.
    """
    parsed: dict[Finger, Requirement] = {}
    if isinstance(spec, str):
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise GestureError(f"expected 'finger:state' in {chunk!r}")
            name, _, state = chunk.partition(":")
            parsed[_as_finger(name.strip())] = _as_requirement(state.strip())
    else:
        for key, value in spec.items():
            finger = key if isinstance(key, Finger) else _as_finger(str(key))
            requirement = value if isinstance(value, Requirement) else _as_requirement(str(value))
            parsed[finger] = requirement

    for finger in FINGERS:
        parsed.setdefault(finger, Requirement.ANY)
    return parsed


def _as_finger(name: str) -> Finger:
    try:
        return Finger(name.strip().lower())
    except ValueError as exc:
        valid = ", ".join(f.value for f in FINGERS)
        raise GestureError(f"unknown finger {name!r}; expected one of: {valid}") from exc


def _as_requirement(state: str) -> Requirement:
    normalised = state.strip().lower()
    aliases = {"up": "up", "down": "down", "any": "any", "-": "any", "ignore": "any"}
    try:
        return Requirement(aliases[normalised])
    except KeyError as exc:
        raise GestureError(
            f"unknown finger state {state!r}; expected 'up', 'down' or 'any'"
        ) from exc


@dataclass(frozen=True, slots=True)
class GestureSpec:
    """One named gesture: what to look for, and what to do about it."""

    name: str
    action: str
    requirements: Mapping[Finger, Requirement]
    label: str = ""
    """Short ASCII text for the HUD, e.g. ``"INDEX"``."""

    description: str = ""
    """Longer explanation, used by ``--list-gestures`` and the README."""

    repeatable: bool | None = None
    """``None`` defers to the action's own repeatability."""

    cooldown: float | None = None
    """``None`` defers to the action's own cooldown."""

    @property
    def constraints(self) -> frozenset[tuple[Finger, Requirement]]:
        """Only the fingers this gesture actually constrains."""
        return frozenset(
            (finger, requirement)
            for finger, requirement in self.requirements.items()
            if requirement is not Requirement.ANY
        )

    @property
    def specificity(self) -> int:
        """How many fingers are constrained; more specific wins a tie."""
        return len(self.constraints)

    @property
    def is_constrained(self) -> bool:
        return self.specificity > 0

    def matches(self, pose: HandPose) -> bool:
        """True when ``pose`` satisfies every constraint of this gesture."""
        return requirements_match(self.requirements, pose)

    # -- presentation ------------------------------------------------------ #

    def requirement_text(self) -> str:
        """Human-readable form, e.g. ``"index: up, middle: down"``."""
        return requirements_text(self.requirements)

    def pose_text(self) -> str:
        """Compact ASCII rendering of the hand shape, thumb first.

        A fist reads ``"v v v v v"`` and index+middle reads ``"v ^ ^ v v"``.
        """
        return requirements_symbols(self.requirements)

    def guide_line(self) -> str:
        """One-line HUD entry, generated from the requirement table."""
        return f"{self.pose_text()}  {self.label or self.name.upper()}"

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "requirements": {
                finger.value: requirement.value
                for finger, requirement in self.requirements.items()
            },
            "action": self.action,
        }
        if self.label:
            payload["label"] = self.label
        if self.description:
            payload["description"] = self.description
        if self.repeatable is not None:
            payload["repeatable"] = self.repeatable
        if self.cooldown is not None:
            payload["cooldown"] = self.cooldown
        return payload


@dataclass
class GestureSet:
    """An ordered, validated collection of gestures.

    Performance note: :meth:`classify` scans every gesture, but each check exits
    on the first mismatching finger, and real sets are ~5-10 entries. A bitmask
    index is deliberately avoided -- it would make the shadowing analysis below
    harder to explain, and profiling shows it is not the bottleneck (the
    detector is, by two orders of magnitude).
    """

    specs: tuple[GestureSpec, ...] = ()
    warnings: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.specs:
            raise GestureError("a gesture set needs at least one gesture")
        self.warnings = tuple(self.warnings) + tuple(self._analyse())

    def __iter__(self) -> Iterator[GestureSpec]:
        return iter(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, index: int) -> GestureSpec:
        return self.specs[index]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def by_name(self, name: str) -> GestureSpec | None:
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

    # -- validation -------------------------------------------------------- #

    def _analyse(self) -> list[str]:
        """Reject genuinely ambiguous sets and report shadowed gestures.

        Two gestures can both match some pose unless they contradict each other
        on a shared finger (one demands UP where the other demands DOWN). So:

        * contradicting gestures are safely disjoint -- nothing to report;
        * overlapping gestures where one constrains strictly more fingers resolve
          deterministically (the specific one wins) but are almost never what the
          author intended, so they are reported;
        * overlapping gestures with *equal* specificity are genuinely ambiguous
          and rejected, because the winner would depend on declaration order.
        """
        problems: list[str] = []
        seen_names: set[str] = set()
        for spec in self.specs:
            if spec.name in seen_names:
                raise GestureError(f"duplicate gesture name {spec.name!r}")
            seen_names.add(spec.name)
            if not spec.is_constrained:
                raise GestureError(
                    f"gesture {spec.name!r} constrains no fingers and would match every pose"
                )

        for index, first in enumerate(self.specs):
            for second in self.specs[index + 1:]:
                if _contradicts(first, second):
                    continue
                if first.constraints == second.constraints:
                    raise GestureError(
                        f"gestures {first.name!r} and {second.name!r} have identical finger "
                        f"requirements, so one could never be chosen over the other"
                    )
                if first.specificity == second.specificity:
                    raise GestureError(
                        f"gestures {first.name!r} and {second.name!r} can both match the same "
                        f"pose and neither is more specific; make one of them constrain more "
                        f"fingers, or separate them with an explicit up/down requirement"
                    )
                winner, loser = (
                    (first, second) if first.specificity > second.specificity else (second, first)
                )
                problems.append(
                    f"{loser.name!r} overlaps {winner.name!r}; {winner.name!r} wins because it "
                    f"constrains {winner.specificity} fingers to {loser.specificity}"
                )
        return problems

    # -- classification ---------------------------------------------------- #

    def classify(self, pose: HandPose | None) -> GestureSpec | None:
        """Return the best-matching gesture, or ``None`` if nothing matches.

        The most specific match wins, so a general catch-all cannot steal a pose
        from a more detailed gesture.
        """
        if pose is None:
            return None
        best: GestureSpec | None = None
        for spec in self.specs:
            if best is not None and spec.specificity <= best.specificity:
                continue
            if spec.matches(pose):
                best = spec
        return best

    def guide_lines(self, width: int | None = None) -> list[str]:
        """HUD-ready guide text, optionally wrapped to ``width`` characters."""
        lines = [spec.guide_line() for spec in self.specs]
        if width is None or width <= 0:
            return lines
        wrapped: list[str] = []
        for line in lines:
            if len(line) <= width:
                wrapped.append(line)
                continue
            head, _, tail = line.rpartition("  ")
            wrapped.append(head)
            wrapped.append(" " * 4 + tail)
        return wrapped


def _contradicts(first: GestureSpec, second: GestureSpec) -> bool:
    """True when two gestures demand opposite states of some shared finger."""
    for finger, requirement in first.requirements.items():
        other = second.requirements.get(finger, Requirement.ANY)
        if requirement is Requirement.ANY or other is Requirement.ANY:
            continue
        if requirement is not other:
            return True
    return False


def build_gesture_set(specs: Sequence[GestureSpec] | Iterable[GestureSpec]) -> GestureSet:
    """Validate and freeze a sequence of gesture specs."""
    return GestureSet(specs=tuple(specs))
