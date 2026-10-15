"""Action interface, registry, and the off-thread executor.

What an "action" is
-------------------
A gesture maps to an action *name* (``"volume_up"``); the registry turns that
name into behaviour. The indirection is what lets the same recogniser drive
volume, brightness, arrow keys or media keys, and what lets ``--dry-run`` verify
a mapping without touching the system.

Why actions run off the render thread
-------------------------------------
Volume and brightness changes are not free: a ``pycaw`` COM call is ~1 ms, but
the Windows brightness path is a WMI round trip that measures in the hundreds of
milliseconds. The original called these inline in the frame loop, so every
volume step cost a stutter in the video and, on a slow path, visibly froze the
preview.

:class:`ActionExecutor` moves them to a worker thread with **per-action
coalescing**: at most one request per action is ever pending, and a new request
replaces an older pending one. That is exactly the right policy for a level
control -- if three volume-up steps arrive while the first is still executing,
the user wants the volume to end up three steps higher, not for three blocking
calls to be replayed later. It also stops a slow backend from building an
unbounded backlog.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping

__all__ = [
    "ActionOutcome",
    "Action",
    "ActionExecutor",
    "ActionRegistry",
    "ActionError",
]

_LOG = logging.getLogger(__name__)


class ActionError(RuntimeError):
    """Raised by an action that cannot be performed at all."""


@dataclass(frozen=True, slots=True)
class ActionOutcome:
    """The result of performing an action, for logging and the HUD."""

    ok: bool
    message: str = ""
    value: float | None = None
    """New level for level-style actions (volume 0-1, brightness 0-100)."""

    detail: str = ""
    """Extra context, e.g. the backend that serviced the request."""

    @property
    def failed(self) -> bool:
        return not self.ok


class Action(ABC):
    """Something a gesture can trigger."""

    #: Registry key referenced from configuration.
    id: str = ""

    #: Short label for the HUD.
    label: str = ""

    #: Whether holding the gesture should repeat this action.
    repeatable: bool = False

    #: Minimum seconds between two firings of *this* action.
    cooldown: float = 0.0

    #: True when the action talks to the operating system.
    touches_system: bool = True

    #: True when this action may still fire while the application is paused.
    #:
    #: This is what makes a pause reversible by gesture. A pause state that
    #: swallows every gesture is a trap: the user pauses, then has to reach for
    #: the keyboard of a machine they were deliberately controlling without one.
    #: Only controls that *manage* the paused state -- resuming, quitting -- set
    #: this, and the application recognises nothing else while paused.
    works_while_paused: bool = False

    # Note: ``id`` is deliberately not validated in a base constructor.
    # Subclasses commonly assign it *after* calling ``super().__init__()``, so a
    # check there would enforce an accidental ordering rule on every subclass.
    # :meth:`ActionRegistry.register` validates each action as it enters the
    # system instead.

    @abstractmethod
    def invoke(self) -> ActionOutcome:
        """Perform the action and report what happened."""

    def describe(self) -> str:
        """One-line description for ``--list-actions`` and the doctor."""
        return f"{self.id}: {self.label or self.id}"

    def teardown(self) -> None:
        """Release any resources held by the action."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} id={self.id!r}>"


class ActionExecutor:
    """Runs actions on a worker thread, coalescing pending requests per action.

    Set ``enabled=False`` to run everything inline, which is what the tests and
    ``--sync-actions`` use: deterministic ordering with no thread involved.
    """

    def __init__(self, *, enabled: bool = True, name: str = "action-worker") -> None:
        self._enabled = enabled
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._name = name
        self._closed = False
        self._runner = None  # set by the registry: callable(action_id) -> outcome
        self.last_outcome: ActionOutcome | None = None
        self.executed = 0
        self.coalesced = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def bind(self, runner) -> None:
        """Attach the callable that actually performs an action."""
        self._runner = runner
        if self._enabled and self._thread is None:
            self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
            self._thread.start()

    def submit(self, action_id: str, *, coalesce: bool = True) -> bool:
        """Queue ``action_id``. Returns True if it was newly queued.

        Returns False when an identical request is already pending and
        ``coalesce`` is set -- the caller can treat that as "already handled".
        """
        if not self._enabled:
            return False
        with self._lock:
            if action_id in self._pending:
                self.coalesced += 1
                return False
            self._pending.add(action_id)
        self._queue.put(action_id)
        return True

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                if self._runner is not None:
                    self.last_outcome = self._runner(item)
                    self.executed += 1
            except Exception:  # pragma: no cover - the runner already guards
                _LOG.exception("action %r raised on the worker thread", item)
            finally:
                with self._lock:
                    self._pending.discard(item)
                self._queue.task_done()

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def drain(self, timeout: float = 2.0) -> bool:
        """Block until every queued action has run. Used by tests and shutdown."""
        if not self._enabled or self._thread is None:
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._pending:
                    return True
            time.sleep(0.001)
        return False

    def close(self, timeout: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is None:
            return
        self.drain(timeout)
        self._queue.put(None)
        self._thread.join(timeout=timeout)
        self._thread = None


class ActionRegistry:
    """Look up actions by id and perform them safely.

    Every failure is contained: an action that raises (a missing COM interface,
    a locked audio device, a keyboard API that refuses to synthesise input)
    produces a failed :class:`ActionOutcome` and a log entry. The original
    printed an error and carried on, which was right; this also surfaces it in
    the HUD so a silently broken gesture is visible rather than mysterious.
    """

    def __init__(
        self,
        actions: Iterable[Action] = (),
        *,
        dry_run: bool = False,
        async_actions: bool = True,
    ) -> None:
        self.dry_run = dry_run
        self._actions: dict[str, Action] = {}
        self.invocations: dict[str, int] = {}
        self.failures: dict[str, int] = {}
        self.last_outcome: Mapping[str, ActionOutcome] = {}
        for action in actions:
            self.register(action)
        self.executor = ActionExecutor(enabled=async_actions and not dry_run)
        self.executor.bind(self._perform)

    # -- registration ------------------------------------------------------ #

    def register(self, action: Action) -> None:
        action_id = getattr(action, "id", "")
        if not isinstance(action_id, str) or not action_id:
            raise ActionError(f"{type(action).__name__} does not define a usable id")
        if action_id in self._actions:
            raise ActionError(f"duplicate action id {action_id!r}")
        self._actions[action_id] = action
        self.invocations[action_id] = 0
        self.failures[action_id] = 0

    def __contains__(self, action_id: object) -> bool:
        return action_id in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._actions)

    def get(self, action_id: str) -> Action | None:
        return self._actions.get(action_id)

    def repeatable(self, action_id: str) -> bool:
        action = self._actions.get(action_id)
        return bool(action and action.repeatable)

    def cooldown(self, action_id: str) -> float:
        action = self._actions.get(action_id)
        return float(action.cooldown) if action else 0.0

    def works_while_paused(self, action_id: str) -> bool:
        """True when the action may fire while the application is paused.

        The pause state stops the controller acting on your machine; it must not
        make the controller impossible to leave. Only the controls that manage the
        paused state advertise this, and the application consults it to decide
        which gestures stay recognisable while paused.
        """
        action = self._actions.get(action_id)
        return bool(action and action.works_while_paused)

    def ids_works_while_paused(self) -> tuple[str, ...]:
        return tuple(
            action_id
            for action_id, action in self._actions.items()
            if action.works_while_paused
        )

    def describe_all(self) -> list[str]:
        return [action.describe() for action in self._actions.values()]

    def labels(self) -> dict[str, str]:
        """Human-readable label per action id.

        Used by the HUD legend, which shows what each gesture *does* in plain
        words rather than making the user decode the action's identifier.
        """
        return {
            action_id: (action.label or action_id)
            for action_id, action in self._actions.items()
        }

    # -- invocation -------------------------------------------------------- #

    def invoke(self, action_id: str, *, blocking: bool | None = None) -> ActionOutcome:
        """Perform an action.

        With async actions enabled and ``blocking`` not set, a *system* action
        returns immediately with an ``ok`` outcome describing the request; the
        real result is recorded in :attr:`last_outcome`.

        Actions that do **not** touch the system -- pausing, changing mode,
        quitting -- always run inline, on the calling thread. They mutate
        application state that the frame loop owns, so running them on the worker
        would be a data race against the very loop that scheduled them, and it
        would also make a mode change take effect a frame or two late.
        """
        action = self._actions.get(action_id)
        if action is None:
            return ActionOutcome(
                ok=False,
                message=f"unknown action {action_id!r}",
                detail=f"known: {', '.join(self._actions)}",
            )

        if self.dry_run:
            self.invocations[action_id] += 1
            outcome = ActionOutcome(ok=True, message="dry run", detail=action.describe())
            self.last_outcome = {**self.last_outcome, action_id: outcome}
            return outcome

        use_async = self.executor.enabled if blocking is None else not blocking
        if use_async and action.touches_system:
            if self.executor.submit(action_id):
                return ActionOutcome(ok=True, message="queued", detail=action.describe())
            # Already pending; the coalesced request will cover it.
            return ActionOutcome(ok=True, message="already pending", detail=action.describe())

        return self._perform(action_id)

    def _perform(self, action_id: str) -> ActionOutcome:
        """Run an action, containing any failure."""
        action = self._actions.get(action_id)
        if action is None:  # pragma: no cover - guarded by invoke
            return ActionOutcome(ok=False, message=f"unknown action {action_id!r}")

        self.invocations[action_id] = self.invocations.get(action_id, 0) + 1
        try:
            outcome = action.invoke()
        except Exception as exc:
            self.failures[action_id] = self.failures.get(action_id, 0) + 1
            _LOG.warning("action %s failed: %s", action_id, exc)
            outcome = ActionOutcome(ok=False, message=str(exc) or type(exc).__name__)
        else:
            if outcome.failed:
                self.failures[action_id] = self.failures.get(action_id, 0) + 1
                _LOG.warning("action %s failed: %s", action_id, outcome.message)
            else:
                _LOG.debug("action %s ok: %s", action_id, outcome.message)

        self.last_outcome = {**self.last_outcome, action_id: outcome}
        return outcome

    # -- lifecycle --------------------------------------------------------- #

    def drain(self, timeout: float = 2.0) -> bool:
        return self.executor.drain(timeout)

    def close(self) -> None:
        self.executor.close()
        for action in self._actions.values():
            try:
                action.teardown()
            except Exception:  # pragma: no cover - defensive
                _LOG.debug("teardown of %s failed", action.id, exc_info=True)

    @property
    def total_failures(self) -> int:
        return sum(self.failures.values())

    @property
    def total_invocations(self) -> int:
        return sum(self.invocations.values())
