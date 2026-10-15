"""Temporal logic: deciding *when* a recognised gesture becomes an action.

The original used this::

    gesture_history.append(gesture)                       # only when truthy
    if len(gesture_history) >= 3 and len(set(gesture_history[-3:])) == 1:
        execute(gesture)

which has three defects:

1. ``None`` frames are never appended, so the history never notices that the
   gesture was interrupted. A three-frame run that happened before an
   interruption still counts afterwards.
2. It is frame-count based, so its real duration depends on the frame rate -- 3
   frames is 0.1 s on a 30 fps camera and 0.5 s on a 6 fps one.
3. Because ``last_action`` is a single global timestamp with a 1 s delay, every
   action is locked out for a second, and holding a gesture can never repeat --
   so raising the volume from 20% to 80% took six separate gestures.

:class:`GestureDebouncer` replaces it with an explicit state machine:

* **Time-based arming** (``arm_seconds``) so behaviour is frame-rate independent.
* **Explicit release** (``release_seconds``) so a dropped frame does not
  interrupt a held gesture, while a genuine stop does.
* **Hold-to-repeat** for actions that can repeat (volume, brightness), with a
  longer initial delay and a shorter repeat interval. A brief gesture is still
  one step; holding ramps smoothly.
* **Per-gesture cooldown**, so a non-repeatable action cannot machine-gun while
  different gestures stay immediately available.

The whole class is driven by an explicit ``now`` argument rather than reading a
clock, which makes every timing rule deterministically testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["Trigger", "DebounceStatus", "GestureDebouncer", "TriggerSource"]


@runtime_checkable
class TriggerSource(Protocol):
    """What the debouncer needs from a gesture.

    A protocol rather than a concrete type, because two kinds of gesture flow
    through here: single-hand :class:`~gesture_control.gestures.GestureSpec` and
    two-hand :class:`~gesture_control.modes.TwoHandGesture`. Both expose these
    attributes, and the debouncer genuinely does not care which it is holding --
    which is what lets mode changes and quitting share the same timing machinery
    as a volume step.
    """

    name: str
    action: str
    label: str
    repeatable: bool | None
    cooldown: float | None


@dataclass(frozen=True, slots=True)
class Trigger:
    """A decision that a gesture's action should fire now."""

    gesture: TriggerSource
    is_repeat: bool
    held_seconds: float

    @property
    def name(self) -> str:
        return self.gesture.name


@dataclass(frozen=True, slots=True)
class DebounceStatus:
    """Snapshot for the HUD, so the user can see why nothing has fired yet."""

    candidate: GestureSpec | None = None
    active: GestureSpec | None = None
    armed_fraction: float = 0.0
    held_seconds: float = 0.0
    repeats_fired: int = 0
    blocked_by_cooldown: bool = False

    @property
    def arming(self) -> bool:
        return self.candidate is not None and self.armed_fraction < 1.0


@dataclass
class _Hold:
    """Bookkeeping for the currently active gesture."""

    spec: GestureSpec
    started: float
    last_fire: float
    repeats: int = 0
    next_repeat: float | None = None


class GestureDebouncer:
    """Turns a stream of candidate gestures into a stream of triggers.

    Args:
        arm_seconds: how long a gesture must be held before it first fires.
            ~0.1 s is about three frames at 30 fps -- the same intent as the
            original's "3 consistent frames", but expressed so it survives a
            change of frame rate.
        release_seconds: how long the gesture must be absent before it counts as
            released. Longer than one frame, so a single detection dropout does
            not break a hold.
        repeat_initial_delay: delay before the first repeat, so a quick gesture
            produces exactly one step.
        repeat_interval: spacing of subsequent repeats.
        default_cooldown: cooldown for gestures that do not specify one.
    """

    def __init__(
        self,
        *,
        arm_seconds: float = 0.10,
        release_seconds: float = 0.12,
        repeat_initial_delay: float = 0.45,
        repeat_interval: float = 0.12,
        default_cooldown: float = 0.25,
    ) -> None:
        if arm_seconds < 0.0 or release_seconds < 0.0:
            raise ValueError("arm_seconds and release_seconds must not be negative")
        if repeat_interval <= 0.0:
            raise ValueError("repeat_interval must be positive")
        self.arm_seconds = float(arm_seconds)
        self.release_seconds = float(release_seconds)
        self.repeat_initial_delay = float(repeat_initial_delay)
        self.repeat_interval = float(repeat_interval)
        self.default_cooldown = float(default_cooldown)

        self._pending: GestureSpec | None = None
        self._pending_since: float | None = None
        self._hold: _Hold | None = None
        self._absent_since: float | None = None
        self._last_seen: float | None = None
        self._last_fire: dict[str, float] = {}
        self._status = DebounceStatus()

    # -- introspection ----------------------------------------------------- #

    @property
    def status(self) -> DebounceStatus:
        return self._status

    @property
    def active(self) -> GestureSpec | None:
        return self._hold.spec if self._hold else None

    @property
    def pending(self) -> GestureSpec | None:
        return self._pending

    # -- lifecycle --------------------------------------------------------- #

    def reset(self) -> None:
        """Forget all state; used when the hand leaves the frame or on resume."""
        self._pending = None
        self._pending_since = None
        self._hold = None
        self._absent_since = None
        self._last_seen = None
        self._last_fire.clear()
        self._status = DebounceStatus()

    # -- helpers ----------------------------------------------------------- #

    @staticmethod
    def _repeatable(spec: GestureSpec) -> bool:
        return bool(spec.repeatable)

    def _cooldown(self, spec: GestureSpec) -> float:
        return self.default_cooldown if spec.cooldown is None else float(spec.cooldown)

    def _on_cooldown(self, spec: GestureSpec, now: float) -> bool:
        last = self._last_fire.get(spec.name)
        return last is not None and (now - last) < self._cooldown(spec)

    def _fire(self, spec: GestureSpec, now: float, held: float, *, repeat: bool) -> Trigger:
        self._last_fire[spec.name] = now
        return Trigger(gesture=spec, is_repeat=repeat, held_seconds=held)

    # -- main entry point -------------------------------------------------- #

    def update(
        self,
        candidate: GestureSpec | None,
        now: float,
        *,
        paused: bool = False,
    ) -> list[Trigger]:
        """Advance the machine and return any triggers produced this frame.

        Args:
            candidate: this frame's recognised gesture, or ``None``.
            now: monotonic timestamp in seconds.
            paused: when true, gesture recognition continues (so the HUD still
                updates) but nothing fires and holds are released.
        """
        now = float(now)

        if paused:
            self._pending = None
            self._pending_since = None
            self._hold = None
            self._absent_since = None
            self._status = DebounceStatus(candidate=None, active=None)
            self._last_seen = now
            return []

        if candidate is None:
            return self._update_absent(now)
        return self._update_present(candidate, now)

    def _update_absent(self, now: float) -> list[Trigger]:
        """No gesture this frame: count toward releasing the active hold."""
        self._pending = None
        self._pending_since = None
        triggers: list[Trigger] = []

        if self._hold is not None:
            if self._absent_since is None:
                self._absent_since = now
            if now - self._absent_since >= self.release_seconds:
                self._hold = None
                self._absent_since = None
        else:
            self._absent_since = None

        self._status = DebounceStatus(
            candidate=None,
            active=self._hold.spec if self._hold else None,
            armed_fraction=0.0,
            held_seconds=(now - self._hold.started) if self._hold else 0.0,
            repeats_fired=self._hold.repeats if self._hold else 0,
        )
        self._last_seen = now
        return triggers

    def _update_present(self, candidate: GestureSpec, now: float) -> list[Trigger]:
        """A gesture is visible: arm it, or service the active hold."""
        self._absent_since = None
        self._last_seen = now
        triggers: list[Trigger] = []

        # Track how long *this* candidate has been the visible gesture. Any
        # change of candidate restarts the clock, which is the behaviour the
        # original's never-resetting deque was missing.
        if self._pending is not None and self._pending.name == candidate.name:
            pending_since = self._pending_since if self._pending_since is not None else now
        else:
            self._pending = candidate
            pending_since = now
        self._pending_since = pending_since
        elapsed = now - pending_since

        if self._hold is not None and self._hold.spec.name == candidate.name:
            triggers.extend(self._service_hold(now))
        elif elapsed >= self.arm_seconds:
            if self._hold is not None:
                # Switching straight from one gesture to another: release the
                # old hold and start the new one in the same frame.
                self._hold = None
            if not self._on_cooldown(candidate, now):
                self._hold = _Hold(
                    spec=candidate,
                    started=now - elapsed,
                    last_fire=now,
                    next_repeat=now + self.repeat_initial_delay
                    if self._repeatable(candidate)
                    else None,
                )
                triggers.append(self._fire(candidate, now, elapsed, repeat=False))

        self._publish_status(candidate, now, elapsed)
        return triggers

    def _service_hold(self, now: float) -> list[Trigger]:
        """Fire repeats for the active hold if it is due."""
        hold = self._hold
        assert hold is not None  # noqa: S101 - internal invariant
        if not self._repeatable(hold.spec):
            return []
        if hold.next_repeat is None or now < hold.next_repeat:
            return []
        if self._on_cooldown(hold.spec, now):
            return []

        hold.repeats += 1
        # Re-base on the scheduled time rather than ``now`` so a stalled frame
        # cannot permanently shift the repeat cadence.
        hold.next_repeat += self.repeat_interval
        if hold.next_repeat < now:
            hold.next_repeat = now + self.repeat_interval
        return [self._fire(hold.spec, now, now - hold.started, repeat=True)]

    def _publish_status(self, candidate: GestureSpec, now: float, elapsed: float) -> None:
        """Record a HUD snapshot describing the current arming/hold state."""
        hold = self._hold
        active = hold.spec if hold else None

        if hold is not None and active is not None and active.name == candidate.name:
            fraction = 1.0
            held = now - hold.started
        elif self.arm_seconds <= 0.0:
            fraction = 1.0
            held = elapsed
        else:
            fraction = max(0.0, min(1.0, elapsed / self.arm_seconds))
            held = elapsed

        self._status = DebounceStatus(
            candidate=candidate,
            active=active,
            armed_fraction=fraction,
            held_seconds=max(0.0, held),
            repeats_fired=hold.repeats if hold else 0,
            blocked_by_cooldown=self._on_cooldown(candidate, now),
        )
