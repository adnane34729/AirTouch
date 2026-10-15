"""Actions that control the application itself.

Pausing, changing mode and quitting are not system calls -- they change the
state of the controller. They are still actions, so that gesture configuration
treats them exactly like *volume up*: same registry, same dry-run behaviour, same
cooldown handling.

Two of them declare :attr:`~gesture_control.actions.base.Action.works_while_paused`:

* ``pause_toggle`` -- so a pause is reversible without the keyboard;
* ``quit_app`` -- so there is always a way out while paused.

Nothing else fires while paused. That is the deliberate part: the pause state
should stop the controller doing things to your machine, not lock you out of the
controller.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .base import Action, ActionOutcome

__all__ = [
    "AppController",
    "PassiveController",
    "PauseToggleAction",
    "ModeStepAction",
    "ModeGotoAction",
    "ModeHomeAction",
    "QuitAction",
    "NoOpAction",
]


@runtime_checkable
class AppController(Protocol):
    """What the application must expose for these actions to work.

    A protocol rather than a concrete import, so the actions package does not
    depend on the application module (which imports the actions package).
    """

    paused: bool

    @property
    def mode_count(self) -> int:
        """Number of configured modes."""

    @property
    def mode_label(self) -> str:
        """Display label of the active mode."""

    def set_paused(self, paused: bool) -> None:
        """Enter or leave the paused state."""

    def step_mode(self, delta: int) -> str:
        """Move ``delta`` modes and return the new mode's label."""

    def goto_mode(self, name: str) -> str:
        """Switch to a named mode and return its label."""

    def home_mode(self) -> str:
        """Switch to the first mode and return its label."""

    def request_quit(self) -> None:
        """Ask the application to shut down."""


class PauseToggleAction(Action):
    """Toggle recognition on and off."""

    id = "pause_toggle"
    label = "Pause / Resume"
    repeatable = False
    cooldown = 0.8
    touches_system = False
    works_while_paused = True

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self.controller = controller

    def invoke(self) -> ActionOutcome:
        target = not self.controller.paused
        self.controller.set_paused(target)
        if target:
            return ActionOutcome(
                ok=True,
                message="PAUSED - make the pause gesture again to resume",
                detail="gestures are ignored until resumed",
            )
        return ActionOutcome(ok=True, message="RESUMED", detail="gestures active")


class ModeStepAction(Action):
    """Move to the next or previous mode."""

    repeatable = False
    cooldown = 0.7
    touches_system = False
    works_while_paused = True

    def __init__(self, controller: AppController, *, delta: int = 1) -> None:
        super().__init__()
        self.controller = controller
        self.delta = delta
        self.id = "mode_next" if delta > 0 else "mode_prev"
        self.label = "Next mode" if delta > 0 else "Previous mode"

    def describe(self) -> str:
        return f"{self.id}: {self.label} ({self.controller.mode_count} mode(s) configured)"

    def invoke(self) -> ActionOutcome:
        if self.controller.mode_count <= 1:
            return ActionOutcome(
                ok=True,
                message="only one mode is configured",
                detail="add modes to use mode switching",
            )
        label = self.controller.step_mode(self.delta)
        return ActionOutcome(
            ok=True,
            message=f"mode: {label}",
            detail=f"mode now {label}",
        )


class ModeGotoAction(Action):
    """Jump straight to a named mode."""

    repeatable = False
    cooldown = 0.7
    touches_system = False
    works_while_paused = True

    def __init__(self, controller: AppController, mode: str) -> None:
        super().__init__()
        self.controller = controller
        self.mode = mode
        self.id = f"mode_goto_{mode}"
        self.label = f"Mode: {mode}"

    def describe(self) -> str:
        return f"{self.id}: switch to mode {self.mode!r}"

    def invoke(self) -> ActionOutcome:
        label = self.controller.goto_mode(self.mode)
        return ActionOutcome(ok=True, message=f"mode: {label}", detail=f"mode now {label}")


class ModeHomeAction(Action):
    """Return to the first configured mode.

    Deliberately not "go to the mode called media": referring to a mode by name
    would break the moment a user renames or reorders their modes, whereas "the
    first one" is always well defined and always exists.
    """

    id = "mode_home"
    label = "First mode"
    repeatable = False
    cooldown = 0.7
    touches_system = False
    works_while_paused = True

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self.controller = controller

    def describe(self) -> str:
        return f"{self.id}: switch back to the first mode"

    def invoke(self) -> ActionOutcome:
        label = self.controller.home_mode()
        return ActionOutcome(ok=True, message=f"mode: {label}", detail=f"mode now {label}")


class QuitAction(Action):
    """Shut the controller down.

    Intended to be bound to a two-hand gesture: quitting is destructive enough
    that it should be impossible to trigger one-handed by accident.
    """

    id = "quit_app"
    label = "Quit"
    repeatable = False
    cooldown = 2.0
    touches_system = False
    works_while_paused = True

    def __init__(self, controller: AppController) -> None:
        super().__init__()
        self.controller = controller

    def invoke(self) -> ActionOutcome:
        self.controller.request_quit()
        return ActionOutcome(ok=True, message="quitting", detail="two-hand quit gesture")


class NoOpAction(Action):
    """Deliberately does nothing.

    Useful for authoring: a shape can be bound to ``noop`` to document that it is
    intentionally unused in a mode, and it lets a mode exist without any
    single-hand behaviour at all.
    """

    id = "noop"
    label = "Nothing"
    repeatable = False
    cooldown = 0.0
    touches_system = False

    def __init__(self) -> None:
        super().__init__()

    def invoke(self) -> ActionOutcome:
        return ActionOutcome(ok=True, message="ignored", detail="bound to noop")


class PassiveController:
    """Stand-in controller for building actions with no application running.

    Used by ``--list-actions`` and the doctor, which need the full action list
    (including the application controls) without starting the frame loop. It also
    makes the control actions testable on their own, without a GUI or a camera.
    """

    def __init__(self, *, mode_count: int = 1, mode_label: str = "-") -> None:
        self.paused = False
        self.mode_count = mode_count
        self.mode_label = mode_label
        self.quit_requested = False
        self.mode_steps: list[int] = []
        self.goto_calls: list[str] = []

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)

    def step_mode(self, delta: int) -> str:
        self.mode_steps.append(delta)
        return self.mode_label

    def goto_mode(self, name: str) -> str:
        self.goto_calls.append(name)
        return name

    def home_mode(self) -> str:
        self.goto_calls.append("<home>")
        return self.mode_label

    def request_quit(self) -> None:
        self.quit_requested = True
