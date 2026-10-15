"""Action registry assembly.

Turning configuration into behaviour happens here:

1. build the system actions (volume, mute, brightness) -- which also probes the
   backends, so probe results are available for ``--doctor``;
2. build the keyboard and media actions;
3. build the application controls (pause, mode switching, quit);
4. link every configured gesture, in every mode, to a real action, failing
   loudly on a typo;
5. fill in each gesture's ``repeatable``/``cooldown`` from its action unless the
   gesture overrides them.

Step 4 is worth calling out. In a dict-driven design a misspelled action name
(``"volume_upright"``) produces a gesture that recognises perfectly and does
nothing at all, with no error anywhere. Here it is a startup failure listing the
names that do exist.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Sequence, TypeVar

from ..config import ActionsConfig, ConfigError
from ..gestures import GestureSpec
from ..modes import Mode, TwoHandGesture
from .backends import BackendCapability
from .base import Action, ActionError, ActionOutcome, ActionRegistry
from .control import (
    AppController,
    ModeGotoAction,
    ModeHomeAction,
    ModeStepAction,
    NoOpAction,
    PassiveController,
    PauseToggleAction,
    QuitAction,
)
from .keyboard import (
    KEY_ACTION_SPECS,
    KeyPressAction,
    MediaKeyAction,
    build_keyboard_actions,
    foreground_window_title,
    primary_modifier_name,
)
from .system import BrightnessAction, MuteAction, VolumeAction, build_system_actions

__all__ = [
    "Action",
    "ActionError",
    "ActionOutcome",
    "ActionRegistry",
    "AppController",
    "BackendCapability",
    "BrightnessAction",
    "KEY_ACTION_SPECS",
    "KeyPressAction",
    "MediaKeyAction",
    "ModeGotoAction",
    "ModeHomeAction",
    "ModeStepAction",
    "MuteAction",
    "NoOpAction",
    "PassiveController",
    "PauseToggleAction",
    "QuitAction",
    "VolumeAction",
    "build_application_actions",
    "build_registry",
    "foreground_window_title",
    "primary_modifier_name",
    "resolve_modes",
    "resolve_timing",
]

#: Any configured gesture-like object: single-hand or two-hand.
GestureLike = TypeVar("GestureLike", GestureSpec, TwoHandGesture)


def build_application_actions(
    controller: AppController | None = None,
    *,
    mode_names: Sequence[str] = (),
) -> list[Action]:
    """The actions that manage the controller itself.

    Args:
        controller: the running application, or ``None`` to build passive
            stand-ins (used by ``--list-actions`` and the doctor, which must list
            every action without starting a frame loop).
        mode_names: modes to generate "jump to mode" actions for.
    """
    target: AppController = controller or PassiveController(
        mode_count=max(1, len(mode_names)), mode_label=mode_names[0] if mode_names else "-"
    )
    actions: list[Action] = [
        PauseToggleAction(target),
        ModeStepAction(target, delta=1),
        ModeStepAction(target, delta=-1),
        ModeHomeAction(target),
        NoOpAction(),
        QuitAction(target),
    ]
    actions.extend(ModeGotoAction(target, name) for name in mode_names)
    return actions


def build_registry(
    config: ActionsConfig,
    *,
    controller: AppController | None = None,
    mode_names: Sequence[str] = (),
    async_actions: bool = True,
) -> tuple[ActionRegistry, dict[str, list[BackendCapability]]]:
    """Create the action registry and report backend capabilities.

    The application controls are always built, so ``--list-actions`` and the
    doctor can show and validate them even with no application running.
    """
    system_actions, capabilities = build_system_actions(config)
    actions: list[Action] = [
        *system_actions,
        *build_keyboard_actions(config),
        *build_application_actions(controller, mode_names=mode_names),
    ]
    registry = ActionRegistry(
        actions,
        dry_run=config.dry_run,
        async_actions=async_actions,
    )
    return registry, capabilities


def resolve_timing(
    gestures: Sequence[GestureLike],
    registry: ActionRegistry,
    *,
    context: str = "gesture",
) -> tuple[GestureLike, ...]:
    """Validate action links and inherit timing from each action.

    Works for single-hand gestures and two-hand gestures alike: both are frozen
    dataclasses exposing ``name``, ``action``, ``repeatable`` and ``cooldown``,
    and both should behave the same way at run time.

    Args:
        gestures: the gestures to resolve.
        registry: the actions they must refer to.
        context: prefix for error messages, e.g. ``"mode 'media'"``.

    Raises:
        ConfigError: if a gesture names an action that does not exist.
    """
    resolved: list[GestureLike] = []
    for spec in gestures:
        action = registry.get(spec.action)
        if action is None:
            # List what this registry actually has, not a hardcoded list: the
            # message stays correct when actions are added or removed.
            raise ConfigError(
                f"{context} {spec.name!r} refers to unknown action {spec.action!r}. "
                f"Valid actions: {', '.join(registry.ids)}"
            )

        repeatable = spec.repeatable
        if repeatable is None:
            repeatable = action.repeatable
        elif repeatable and not action.repeatable:
            # Harmless but almost certainly unintended: the gesture will repeat
            # an action that does nothing on the second press.
            logging.getLogger(__name__).warning(
                "%s %r is marked repeatable but action %r is not; "
                "repeats will have no effect",
                context,
                spec.name,
                spec.action,
            )

        cooldown = spec.cooldown
        if cooldown is None:
            # Take the action's value verbatim, including 0.0. Collapsing 0.0 to
            # None here would let it fall back to the debouncer's default
            # cooldown, which silently throttles hold-to-repeat: volume would
            # step at the default 0.25 s instead of the configured 0.12 s.
            cooldown = action.cooldown

        resolved.append(
            dataclasses.replace(spec, repeatable=bool(repeatable), cooldown=cooldown)
        )
    return tuple(resolved)


def resolve_modes(modes: Sequence[Mode], registry: ActionRegistry) -> tuple[Mode, ...]:
    """Validate every mode's gestures against the registry.

    Each mode is resolved independently, and the resulting gesture validation
    happens when the mode's gesture set is built, so a bad table in any mode
    fails at startup with the mode named.
    """
    resolved: list[Mode] = []
    for mode in modes:
        gestures = resolve_timing(
            mode.gestures, registry, context=f"mode {mode.name!r} gesture"
        )
        resolved.append(dataclasses.replace(mode, gestures=gestures))
    return tuple(resolved)
