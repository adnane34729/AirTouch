"""Tests for the action layer: registry isolation, threading, and real actions.

The system-touching actions are exercised against fake backends so the tests are
deterministic and never change the machine's actual volume or brightness.
"""

from __future__ import annotations

import threading
import time
import unittest

from gesture_control.actions.base import (
    Action,
    ActionError,
    ActionOutcome,
    ActionRegistry,
)
from gesture_control.actions.system import BrightnessAction, MuteAction, VolumeAction
from gesture_control.config import ActionsConfig, default_config


class FakeVolumeBackend:
    """In-memory volume backend."""

    name = "fake-volume"
    supports_mute = True

    def __init__(self, level: float = 0.5, *, available: bool = True) -> None:
        self.level = level
        self.muted = False
        self.available = available
        self.writes: list[float] = []
        from gesture_control.actions.backends import BackendCapability

        self.capability = (
            BackendCapability(self.name, True, detail="fake")
            if available
            else BackendCapability(self.name, False, reason="deliberately unavailable")
        )

    def get_percent(self) -> float | None:
        return self.level

    def set_percent(self, value: float) -> None:
        self.level = value
        self.writes.append(value)

    def get_muted(self) -> bool | None:
        return self.muted

    def set_muted(self, muted: bool) -> None:
        self.muted = muted


class FakeBrightnessBackend(FakeVolumeBackend):
    name = "fake-brightness"
    supports_mute = False

    def __init__(self, level: float = 50.0, *, available: bool = True) -> None:
        super().__init__(level, available=available)


class CountAction(Action):
    """An action that counts invocations, optionally raising or sleeping."""

    repeatable = True
    cooldown = 0.0

    def __init__(self, action_id: str = "count", *, fail: bool = False,
                 delay: float = 0.0, blocker: threading.Event | None = None) -> None:
        self.id = action_id
        self.label = action_id
        self.count = 0
        self.fail = fail
        self.delay = delay
        self.blocker = blocker
        self.lock = threading.Lock()
        self.concurrent = 0
        self.max_concurrent = 0
        self.thread_ids: list[int] = []

    def invoke(self) -> ActionOutcome:
        with self.lock:
            self.count += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.thread_ids.append(threading.get_ident())
        try:
            if self.blocker is not None:
                self.blocker.wait(timeout=2.0)
            if self.delay:
                time.sleep(self.delay)
            if self.fail:
                raise RuntimeError("deliberate failure")
            return ActionOutcome(ok=True, message=f"ran {self.count}")
        finally:
            with self.lock:
                self.concurrent -= 1


class TestRegistry(unittest.TestCase):
    def test_register_and_lookup(self) -> None:
        registry = ActionRegistry([CountAction("a"), CountAction("b")], async_actions=False)
        self.assertIn("a", registry)
        self.assertEqual(len(registry), 2)
        self.assertEqual(registry.ids, ("a", "b"))

    def test_duplicate_id_rejected(self) -> None:
        registry = ActionRegistry([CountAction("a")], async_actions=False)
        with self.assertRaises(ActionError):
            registry.register(CountAction("a"))

    def test_action_without_id_rejected(self) -> None:
        class Nameless(Action):
            id = ""

            def invoke(self) -> ActionOutcome:
                return ActionOutcome(ok=True)

        with self.assertRaises(ActionError):
            ActionRegistry([Nameless()], async_actions=False)

    def test_unknown_action_reports_available_ids(self) -> None:
        registry = ActionRegistry([CountAction("a")], async_actions=False)
        outcome = registry.invoke("nope", blocking=True)
        self.assertFalse(outcome.ok)
        self.assertIn("unknown action", outcome.message)
        self.assertIn("a", outcome.detail)

    def test_successful_invocation_counts(self) -> None:
        action = CountAction("a")
        registry = ActionRegistry([action], async_actions=False)
        outcome = registry.invoke("a", blocking=True)
        self.assertTrue(outcome.ok)
        self.assertEqual(action.count, 1)
        self.assertEqual(registry.invocations["a"], 1)
        self.assertEqual(registry.total_failures, 0)

    def test_failure_is_contained(self) -> None:
        """A raising action must not propagate; the loop has to survive it."""
        action = CountAction("boom", fail=True)
        registry = ActionRegistry([action], async_actions=False)
        outcome = registry.invoke("boom", blocking=True)
        self.assertFalse(outcome.ok)
        self.assertIn("deliberate failure", outcome.message)
        self.assertEqual(registry.failures["boom"], 1)
        self.assertEqual(registry.total_failures, 1)

    def test_failing_action_does_not_stop_others(self) -> None:
        registry = ActionRegistry(
            [CountAction("boom", fail=True), CountAction("fine")], async_actions=False
        )
        registry.invoke("boom", blocking=True)
        outcome = registry.invoke("fine", blocking=True)
        self.assertTrue(outcome.ok)

    def test_dry_run_does_not_invoke(self) -> None:
        """Dry run must be airtight: no side effect may occur."""
        action = CountAction("a")
        registry = ActionRegistry([action], dry_run=True, async_actions=False)
        outcome = registry.invoke("a", blocking=True)
        self.assertTrue(outcome.ok)
        self.assertEqual(action.count, 0, "dry run invoked a real action")
        self.assertIn("dry run", outcome.message)

    def test_repeatable_and_cooldown_are_exposed(self) -> None:
        registry = ActionRegistry(
            [VolumeAction(FakeVolumeBackend(), 0.05)], async_actions=False
        )
        self.assertTrue(registry.repeatable("volume_up"))
        self.assertEqual(registry.cooldown("volume_up"), 0.0)


class TestExecutor(unittest.TestCase):
    def test_async_actions_run_on_a_worker(self) -> None:
        action = CountAction("a")
        registry = ActionRegistry([action], async_actions=True)
        try:
            main_thread = threading.get_ident()
            registry.invoke("a")
            self.assertTrue(registry.drain(2.0))
            self.assertEqual(action.count, 1)
            self.assertTrue(action.thread_ids)
            self.assertNotEqual(
                action.thread_ids[0], main_thread, "the action ran on the caller's thread"
            )
        finally:
            registry.close()

    def test_invoke_returns_immediately_when_blocked(self) -> None:
        """The whole point: a slow action must not stall the frame loop."""
        blocker = threading.Event()
        action = CountAction("slow", blocker=blocker)
        registry = ActionRegistry([action], async_actions=True)
        try:
            started = time.perf_counter()
            outcome = registry.invoke("slow")
            elapsed = time.perf_counter() - started
            self.assertTrue(outcome.ok)
            self.assertLess(elapsed, 0.2, "invoke blocked the caller")
            self.assertEqual(outcome.message, "queued")
            blocker.set()
            self.assertTrue(registry.drain(2.0))
            self.assertEqual(action.count, 1)
        finally:
            blocker.set()
            registry.close()

    def test_repeats_are_coalesced_not_queued(self) -> None:
        """A backlog of level changes must collapse to the latest request.

        Otherwise a slow backend accumulates a queue of volume steps that keep
        executing long after the user stopped gesturing.
        """
        blocker = threading.Event()
        action = CountAction("slow", blocker=blocker)
        registry = ActionRegistry([action], async_actions=True)
        try:
            first = registry.invoke("slow")
            self.assertEqual(first.message, "queued")
            for _ in range(20):
                outcome = registry.invoke("slow")
                self.assertIn(outcome.message, {"already pending", "queued"})
            self.assertLessEqual(registry.executor.pending, 1, "requests were not coalesced")
            blocker.set()
            self.assertTrue(registry.drain(2.0))
            # One executing plus at most one coalesced replacement.
            self.assertLessEqual(action.count, 2, f"{action.count} invocations got through")
        finally:
            blocker.set()
            registry.close()

    def test_only_one_action_runs_at_a_time(self) -> None:
        actions = [CountAction(f"a{i}", delay=0.01) for i in range(4)]
        registry = ActionRegistry(actions, async_actions=True)
        try:
            for action in actions:
                registry.invoke(action.id)
            self.assertTrue(registry.drain(3.0))
            self.assertTrue(all(action.max_concurrent == 1 for action in actions))
        finally:
            registry.close()

    def test_shutdown_is_prompt_and_idempotent(self) -> None:
        registry = ActionRegistry([CountAction("a")], async_actions=True)
        registry.invoke("a")
        registry.close()
        registry.close()  # must not raise
        self.assertEqual(registry.executor.executed, 1)

    def test_drain_returns_false_when_something_is_stuck(self) -> None:
        blocker = threading.Event()
        action = CountAction("stuck", blocker=blocker)
        registry = ActionRegistry([action], async_actions=True)
        try:
            registry.invoke("stuck")
            self.assertFalse(registry.drain(0.05))
        finally:
            blocker.set()
            registry.close()


class TestVolumeAction(unittest.TestCase):
    def test_step_up_and_down(self) -> None:
        backend = FakeVolumeBackend(0.5)
        up = VolumeAction(backend, 0.05, direction=1)
        down = VolumeAction(backend, 0.05, direction=-1)
        self.assertTrue(up.invoke().ok)
        self.assertAlmostEqual(backend.level, 0.55, places=6)
        self.assertTrue(down.invoke().ok)
        self.assertAlmostEqual(backend.level, 0.50, places=6)

    def test_clamped_at_the_limits(self) -> None:
        backend = FakeVolumeBackend(0.98)
        up = VolumeAction(backend, 0.05, direction=1)
        outcome = up.invoke()
        self.assertTrue(outcome.ok)
        self.assertAlmostEqual(backend.level, 1.0, places=6)

        backend.level = 0.02
        down = VolumeAction(backend, 0.05, direction=-1)
        down.invoke()
        self.assertAlmostEqual(backend.level, 0.0, places=6)

    def test_already_at_limit_is_not_an_error(self) -> None:
        backend = FakeVolumeBackend(1.0)
        outcome = VolumeAction(backend, 0.05, direction=1).invoke()
        self.assertTrue(outcome.ok)
        self.assertIn("already at", outcome.message)
        self.assertEqual(backend.writes, [], "a no-op step still wrote to the device")

    def test_outcome_reports_the_new_level(self) -> None:
        backend = FakeVolumeBackend(0.4)
        outcome = VolumeAction(backend, 0.1, direction=1).invoke()
        self.assertAlmostEqual(outcome.value, 0.5, places=6)

    def test_unavailable_backend_fails_clearly(self) -> None:
        backend = FakeVolumeBackend(0.5, available=False)
        outcome = VolumeAction(backend, 0.05, direction=1).invoke()
        self.assertFalse(outcome.ok)
        self.assertIn("unavailable", outcome.message)
        self.assertIn("deliberately unavailable", outcome.detail)

    def test_unreadable_level_is_reported(self) -> None:
        backend = FakeVolumeBackend(0.5)
        backend.get_percent = lambda: None  # type: ignore[method-assign]
        outcome = VolumeAction(backend, 0.05, direction=1).invoke()
        self.assertFalse(outcome.ok)
        self.assertIn("could not read", outcome.message)

    def test_invalid_direction_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VolumeAction(FakeVolumeBackend(), 0.05, direction=0)


class TestMuteAction(unittest.TestCase):
    def test_toggles(self) -> None:
        backend = FakeVolumeBackend()
        action = MuteAction(backend)
        action.invoke()
        self.assertTrue(backend.muted)
        action.invoke()
        self.assertFalse(backend.muted)

    def test_unsupported_backend_reports(self) -> None:
        backend = FakeVolumeBackend()
        backend.supports_mute = False
        outcome = MuteAction(backend).invoke()
        self.assertFalse(outcome.ok)
        self.assertIn("cannot control mute", outcome.message)

    def test_unavailable_backend_reports(self) -> None:
        outcome = MuteAction(FakeVolumeBackend(available=False)).invoke()
        self.assertFalse(outcome.ok)
        self.assertIn("unavailable", outcome.message)


class TestBrightnessAction(unittest.TestCase):
    def test_step_and_clamp(self) -> None:
        backend = FakeBrightnessBackend(50.0)
        up = BrightnessAction(backend, 8, direction=1)
        self.assertTrue(up.invoke().ok)
        self.assertAlmostEqual(backend.level, 58.0, places=6)

        backend.level = 98.0
        outcome = up.invoke()
        self.assertTrue(outcome.ok)
        self.assertAlmostEqual(backend.level, 100.0, places=6)

    def test_integer_step_is_preserved(self) -> None:
        backend = FakeBrightnessBackend(40.0)
        BrightnessAction(backend, 8, direction=1).invoke()
        self.assertEqual(backend.writes[-1], 48.0)
        self.assertEqual(float(backend.writes[-1]).is_integer(), True)


class TestActionConstruction(unittest.TestCase):
    def test_every_preset_resolves(self) -> None:
        """Every gesture in every mode, and every two-hand gesture, links to a real action."""
        from gesture_control.actions import build_registry, resolve_modes, resolve_timing
        from gesture_control.config import PRESETS

        for preset in PRESETS:
            with self.subTest(preset=preset):
                config = default_config(preset)
                registry, _ = build_registry(
                    config.actions, mode_names=config.mode_names, async_actions=False
                )
                resolved_modes = resolve_modes(config.modes, registry)
                resolve_timing(config.two_hand, registry, context="two-hand gesture")
                for mode in resolved_modes:
                    for spec in mode.gestures:
                        self.assertIn(spec.action, registry)
                        self.assertIsNotNone(
                            spec.repeatable, f"{preset}/{mode.name}/{spec.name} repeatable unresolved"
                        )
                        self.assertIsNotNone(
                            spec.cooldown, f"{preset}/{mode.name}/{spec.name} cooldown unresolved"
                        )

    def test_repeatable_gestures_match_repeatable_actions(self) -> None:
        """Otherwise holding a gesture silently does nothing after the first step.

        This caught a real mismatch: 'left' was marked repeatable while the
        arrow-key action was not, so the repeats were dropped.
        """
        from gesture_control.actions import build_registry, resolve_modes, resolve_timing
        from gesture_control.config import PRESETS

        for preset in PRESETS:
            config = default_config(preset)
            registry, _ = build_registry(
                config.actions, mode_names=config.mode_names, async_actions=False
            )
            gestures = [
                spec
                for mode in resolve_modes(config.modes, registry)
                for spec in mode.gestures
            ]
            gestures.extend(resolve_timing(config.two_hand, registry))
            for spec in gestures:
                if spec.repeatable:
                    action = registry.get(spec.action)
                    assert action is not None
                    with self.subTest(preset=preset, gesture=spec.name):
                        self.assertTrue(
                            action.repeatable,
                            f"{preset}/{spec.name} is repeatable but {spec.action} is not",
                        )

    def test_unknown_action_name_is_a_config_error(self) -> None:
        from gesture_control.actions import resolve_timing
        from gesture_control.config import ConfigError
        from gesture_control.gestures import GestureSpec, parse_requirements

        registry = ActionRegistry([CountAction("count")], async_actions=False)
        bad = GestureSpec(name="bad", action="nope", requirements=parse_requirements("index:up"))
        with self.assertRaises(ConfigError) as ctx:
            resolve_timing((bad,), registry)
        self.assertIn("unknown action", str(ctx.exception))
        self.assertIn("count", str(ctx.exception))

    def test_timing_resolution_reports_the_context(self) -> None:
        """The error must say *which* mode's gesture is broken."""
        from gesture_control.actions import build_registry, resolve_modes
        from gesture_control.config import ConfigError, build_modes

        registry, _ = build_registry(ActionsConfig(), async_actions=False)
        modes = build_modes(
            [{"name": "broken", "gestures": [
                {"name": "g", "action": "not_a_real_action", "requirements": "index:up"}
            ]}]
        )
        with self.assertRaises(ConfigError) as ctx:
            resolve_modes(modes, registry)
        self.assertIn("broken", str(ctx.exception))

    def test_volume_cooldown_is_not_replaced_by_a_default(self) -> None:
        """A zero cooldown must survive resolution.

        Collapsing 0.0 to None would let it fall back to the debouncer's default
        0.25 s, throttling hold-to-repeat to half the configured rate.
        """
        from gesture_control.actions import build_registry, resolve_modes

        config = default_config("classic")
        registry, _ = build_registry(
            config.actions, mode_names=config.mode_names, async_actions=False
        )
        mode = resolve_modes(config.modes, registry)[0]
        resolved = {spec.name: spec for spec in mode.gestures}
        self.assertEqual(resolved["volume_up"].cooldown, 0.0)

    def test_dry_run_config_reaches_the_registry(self) -> None:
        from gesture_control.actions import build_registry

        config = ActionsConfig(dry_run=True)
        registry, _ = build_registry(config, async_actions=False)
        self.assertTrue(registry.dry_run)


class TestApplicationControlActions(unittest.TestCase):
    """The pause / mode / quit actions, against a passive controller."""

    def _registry(self, controller=None, modes=("a", "b")):
        from gesture_control.actions import build_registry

        config = ActionsConfig()
        registry, _ = build_registry(
            config, controller=controller, mode_names=list(modes), async_actions=False
        )
        return registry

    def test_pause_toggles_and_works_while_paused(self) -> None:
        from gesture_control.actions import PassiveController

        controller = PassiveController()
        registry = self._registry(controller)
        # The whole point of the flag: the pause gesture must still be recognised
        # while paused, so a pause is reversible without the keyboard.
        self.assertTrue(registry.works_while_paused("pause_toggle"))

        outcome = registry.invoke("pause_toggle", blocking=True)
        self.assertTrue(outcome.ok)
        self.assertTrue(controller.paused)
        registry.invoke("pause_toggle", blocking=True)
        self.assertFalse(controller.paused)

    def test_quit_works_while_paused(self) -> None:
        """There must always be a way out of the paused state."""
        from gesture_control.actions import PassiveController

        controller = PassiveController()
        registry = self._registry(controller)
        self.assertTrue(registry.works_while_paused("quit_app"))
        registry.invoke("quit_app", blocking=True)
        self.assertTrue(controller.quit_requested)

    def test_mode_actions_step_and_go_home(self) -> None:
        from gesture_control.actions import PassiveController

        controller = PassiveController(mode_count=3, mode_label="B")
        registry = self._registry(controller, modes=("a", "b", "c"))
        registry.invoke("mode_next", blocking=True)
        registry.invoke("mode_prev", blocking=True)
        registry.invoke("mode_home", blocking=True)
        self.assertEqual(controller.mode_steps, [1, -1])
        self.assertEqual(controller.goto_calls, ["<home>"])

    def test_mode_actions_are_inert_with_one_mode(self) -> None:
        from gesture_control.actions import PassiveController

        controller = PassiveController(mode_count=1)
        registry = self._registry(controller, modes=("only",))
        outcome = registry.invoke("mode_next", blocking=True)
        self.assertTrue(outcome.ok)
        self.assertIn("only one mode", outcome.message)
        self.assertEqual(controller.mode_steps, [])

    def test_system_actions_do_not_work_while_paused(self) -> None:
        """A pause must stop the controller acting on the machine."""
        registry = self._registry()
        for action_id in ("volume_up", "volume_down", "brightness_up", "key_space", "next_track"):
            with self.subTest(action=action_id):
                self.assertFalse(registry.works_while_paused(action_id))

    def test_noop_does_nothing(self) -> None:
        registry = self._registry()
        outcome = registry.invoke("noop", blocking=True)
        self.assertTrue(outcome.ok)
        self.assertIn("ignored", outcome.message)

    def test_application_actions_run_inline_even_with_async_enabled(self) -> None:
        """They mutate state the frame loop owns, so they must not be offloaded.

        Offloading them was a real data race: the worker changed the mode while
        the main thread was classifying gestures against it.
        """
        from gesture_control.actions import ActionRegistry, PassiveController, build_application_actions

        controller = PassiveController(mode_count=2, mode_label="A")
        registry = ActionRegistry(
            build_application_actions(controller, mode_names=["a", "b"]),
            async_actions=True,
        )
        try:
            outcome = registry.invoke("pause_toggle")
            self.assertNotEqual(outcome.message, "queued", "a control action was offloaded")
            self.assertTrue(controller.paused, "the pause did not take effect synchronously")
        finally:
            registry.close()

    def test_passive_controller_is_a_valid_controller(self) -> None:
        from gesture_control.actions import AppController, PassiveController

        self.assertIsInstance(PassiveController(), AppController)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
