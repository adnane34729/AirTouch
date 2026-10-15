"""End-to-end tests for modes, two-hand gestures and gesture-driven pause.

These drive the real application pipeline. The configurations used here map every
mode gesture to ``noop`` or to an application control, so actions genuinely
execute (not dry-run -- mode switching has to actually happen) while nothing on
the machine is touched.
"""

from __future__ import annotations

import unittest

import numpy as np

from gesture_control.app import GestureApp
from gesture_control.config import AppConfig, build_modes, build_two_hand_gestures
from gesture_control.detector import SyntheticHandDetector
from gesture_control.geometry import make_observation
from gesture_control.sources import TestPatternSource
from gesture_control.synthetic import GestureScript, QUIT_SCRIPT, TWO_HAND_SCRIPT
from gesture_control.types import FINGERS, Finger, Handedness
from gesture_control.vocabulary import SHAPES

try:
    import cv2  # noqa: F401  (capability probe)

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False


def virtual_clock(fps: float = 30.0):
    state = {"t": 0.0}

    def clock() -> float:
        value = state["t"]
        state["t"] += 1.0 / fps
        return value

    return clock


def harmless_config(*, single_gestures=("index", "peace"), with_mode_gestures=True):
    """A two-mode config whose every gesture is side-effect free."""
    mode_entries = []
    for name, label in (("one", "ONE"), ("two", "TWO")):
        gestures = [
            {
                "name": f"g{i}",
                "action": "noop",
                "requirements": SHAPES[shape_name],
                "label": shape_name.upper(),
            }
            for i, shape_name in enumerate(single_gestures)
        ]
        mode_entries.append({"name": name, "label": label, "gestures": gestures})

    two_hand = []
    if with_mode_gestures:
        two_hand = [
            {
                "name": "mode_next",
                "action": "mode_next",
                "shape": SHAPES["open"],
                "label": "BOTH OPEN",
                "repeatable": False,
            },
            {
                "name": "mode_prev",
                "action": "mode_prev",
                "shape": SHAPES["fist"],
                "label": "BOTH FISTS",
                "repeatable": False,
            },
            {
                "name": "mode_home",
                "action": "mode_home",
                "shape": SHAPES["peace"],
                "label": "BOTH PEACE",
                "repeatable": False,
            },
        ]
    two_hand.append(
        {
            "name": "pause",
            "action": "pause_toggle",
            "shape": SHAPES["shaka"],
            "label": "BOTH SHAKA",
            "repeatable": False,
        }
    )
    two_hand.append(
        {
            "name": "quit",
            "action": "quit_app",
            "shape": SHAPES["horns"],
            "label": "BOTH HORNS",
            "repeatable": False,
        }
    )

    config = AppConfig(
        preset="pro",
        modes=build_modes(mode_entries),
        two_hand=build_two_hand_gestures(two_hand),
    )
    config.overlay.show_window = False
    config.actions.dry_run = False
    config.actions.allow_keyboard = False
    return config


def build_app(config, script, *, frames_handler=None):
    return GestureApp(
        config,
        source=TestPatternSource(640, 360, fps=0.0),
        detector=SyntheticHandDetector(GestureScript(script, rotation_span=0.0)),
        clock=virtual_clock(),
    )


class ScriptedHands:
    """A detector that plays an exact list of hand poses, one per call."""

    name = "scripted-hands"

    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.stats = None

    def detect(self, frame, timestamp):
        if self.index >= len(self.frames):
            return []
        poses = self.frames[self.index]
        self.index += 1
        observations = []
        for slot, (curls, thumb_out, hand) in enumerate(poses):
            raw = _hand_landmarks(curls, thumb_out, slot)
            observations.append(
                make_observation(
                    raw,
                    aspect_ratio=16 / 9,
                    handedness=hand,
                    handedness_score=1.0,
                    timestamp=timestamp,
                )
            )
        return observations

    def describe(self):
        return "scripted-hands"

    def close(self):
        pass


_CENTER = {0: (0.5, 0.86), 1: (0.2, 0.88)}


def _hand_landmarks(curls, thumb_out, slot):
    from gesture_control.synthetic import make_hand

    return make_hand(
        curls,
        thumb_out=thumb_out,
        scale=0.26,
        center=_CENTER[slot],
        aspect_ratio=16 / 9,
        mirror=slot == 1,
    )


def up_shape(names) -> dict:
    wanted = set(names)
    return {finger: (0.0 if finger in wanted else 1.0) for finger in FINGERS}


def hands(names, thumb_out=0.0):
    """One frame with two hands making the same shape."""
    curls = up_shape(names)
    return (
        (curls, thumb_out, Handedness.RIGHT),
        (curls, thumb_out, Handedness.LEFT),
    )


def single(names, thumb_out=0.0):
    return ((up_shape(names), thumb_out, Handedness.RIGHT),)


class TestModeSwitching(unittest.TestCase):
    def _app(self, frames):
        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(640, 360, fps=0.0),
            detector=ScriptedHands(frames),
            clock=virtual_clock(),
        )
        return app

    def _run(self, frames):
        app = self._app(frames)
        logs: list[tuple[str, str]] = []
        original = app._dispatch

        def spy(trigger, now):
            outcome = original(trigger, now)
            logs.append((trigger.name, app.mode_label))
            return outcome

        app._dispatch = spy
        step = 1.0 / 30.0
        now = 0.0
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        for _ in range(len(frames)):
            app.process(frame, now)
            now += step
        app.shutdown()
        return logs, app

    def test_two_hand_open_advances_the_mode(self) -> None:
        """Both palms held: the mode moves forward, and synchronously."""
        frames = [single([Finger.INDEX])] * 5 + [hands(FINGERS, thumb_out=1.0)] * 30
        logs, app = self._run(frames)
        self.assertEqual(app.mode_label, "TWO")
        self.assertEqual(app._mode_switches, 1)
        mode_next = [entry for entry in logs if entry[0] == "mode_next"]
        self.assertTrue(mode_next)
        self.assertEqual(mode_next[0][1], "TWO", "the mode did not change immediately")

    def test_two_hand_fist_goes_back(self) -> None:
        frames = [hands(FINGERS, thumb_out=1.0)] * 30 + [single([Finger.INDEX])] * 5
        _, app = self._run(frames)
        self.assertEqual(app.mode_label, "TWO")
        # Now go back with both fists.
        app2_frames = (
            [hands(FINGERS, thumb_out=1.0)] * 30
            + [single([Finger.INDEX])] * 4
            + [hands([], thumb_out=0.0)] * 30
        )
        logs, app2 = self._run(app2_frames)
        names = [name for name, _ in logs]
        self.assertIn("mode_next", names)
        self.assertIn("mode_prev", names)
        self.assertEqual(app2.mode_label, "ONE")

    def test_two_hand_script_cycles_modes(self) -> None:
        """The whole two-hand routine, through the real synthetic detector.

        Uses the scripted model rather than hand-built frames, so the hand
        geometry, the mirroring of the second hand and the transition frames are
        all exercised rather than assumed.
        """
        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(640, 360, fps=0.0),
            detector=SyntheticHandDetector(
                GestureScript(TWO_HAND_SCRIPT, rotation_span=0.0)
            ),
            clock=virtual_clock(),
        )
        summary = app.run(max_frames=240)
        self.assertGreater(summary.frames, 100)
        # BOTH OPEN, BOTH PEACE and BOTH FISTS all appear in the routine, so the
        # mode must have moved more than once.
        self.assertGreaterEqual(app._mode_switches, 2, "the script did not switch modes")
        self.assertIn(app.mode_label, ("ONE", "TWO"))

    def test_one_hand_never_changes_the_mode(self) -> None:
        """The safety property, end to end: a mode change needs both hands."""
        frames = [single(FINGERS, thumb_out=1.0)] * 60
        logs, app = self._run(frames)
        self.assertEqual(app._mode_switches, 0)
        self.assertEqual([name for name, _ in logs if name.startswith("mode_")], [])

    def test_only_the_primary_hand_fires_while_both_hands_are_visible(self) -> None:
        """Two hands in frame means a two-hand gesture is being attempted.

        The default policy lets only the configured primary hand drive
        single-hand gestures, so a pose made *only* by the secondary hand must not
        fire, and a pose made by the primary hand may.
        """
        config = harmless_config(single_gestures=("index", "pinky"))
        config.switching.primary_hand = "right"

        def pair(left_up, right_up):
            return (
                (up_shape(left_up), 0.0, Handedness.LEFT),
                (up_shape(right_up), 0.0, Handedness.RIGHT),
            )

        # Only the left (secondary) hand makes a configured single-hand shape.
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands([pair([Finger.INDEX], [])] * 20),
            clock=virtual_clock(),
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        triggers = []
        for step in range(20):
            triggers.extend(app.process(frame, step / 30.0))
        app.shutdown()
        self.assertEqual(triggers, [], "the secondary hand fired a single-hand gesture")

        # Only the right (primary) hand does.
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands([pair([], [Finger.INDEX])] * 20),
            clock=virtual_clock(),
        )
        triggers = []
        for step in range(20):
            triggers.extend(app.process(frame, step / 30.0))
        app.shutdown()
        self.assertTrue(triggers, "the primary hand was ignored")

    def test_switching_suppresses_single_hand_gestures_while_two_visible(self) -> None:
        config = harmless_config()
        config.switching.single_hand_with_two_visible = "suppress"
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands([hands([Finger.INDEX], thumb_out=0.0)] * 20),
            clock=virtual_clock(),
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        triggers = []
        for step in range(20):
            triggers.extend(app.process(frame, step / 30.0))
        app.shutdown()
        self.assertEqual(triggers, [])

    def test_primary_hand_policy_picks_the_configured_hand(self) -> None:
        config = harmless_config()
        config.switching.single_hand_with_two_visible = "primary"
        config.switching.primary_hand = "right"
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands(
                [
                    (
                        (up_shape([Finger.INDEX]), 0.0, Handedness.LEFT),
                        (up_shape([Finger.PINKY]), 0.0, Handedness.RIGHT),
                    )
                ]
                * 5
            ),
            clock=virtual_clock(),
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        app.process(frame, 0.0)
        pose = app._primary_pose()
        app.shutdown()
        self.assertIsNotNone(pose)
        # The right hand is the primary, so the pinky is what single-hand
        # gestures will be classified from.
        self.assertTrue(pose.is_extended(Finger.PINKY))
        self.assertFalse(pose.is_extended(Finger.INDEX))


class TestGesturePause(unittest.TestCase):
    """Request 1: a pause must be escapable by gesture."""

    def _app(self, frames):
        config = harmless_config()
        return GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands(frames),
            clock=virtual_clock(),
        )

    def _drive(self, app, frames_count, start=0.0):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for step in range(frames_count):
            app.process(frame, start + step / 30.0)

    def test_two_hand_shaka_pauses(self) -> None:
        app = self._app([hands([Finger.THUMB, Finger.PINKY], thumb_out=1.0)] * 30)
        self._drive(app, 30)
        app.shutdown()
        self.assertTrue(app.paused)

    def test_the_same_gesture_resumes(self) -> None:
        """The regression that started this: you could pause and not get out.

        The gesture has to be *released* between toggles -- a one-shot gesture
        fires once per hold, which is what stops a held pose from flipping the
        paused state back and forth.
        """
        # ``hands(...)`` is one *frame* holding two hands, so each use needs the
        # list wrapper before repeating it.
        paused_frames = [hands([Finger.THUMB, Finger.PINKY], thumb_out=1.0)] * 30
        # The release has to be a shape that is *no* two-hand gesture, and it has
        # to last longer than the two-hand release tolerance. Both fists would be
        # wrong here: that is the "previous mode" gesture, so the pause hold would
        # never be considered released at all.
        released = [hands([Finger.INDEX])] * 14
        frames = paused_frames + released + paused_frames
        app = self._app(frames)

        self._drive(app, len(paused_frames))
        self.assertTrue(app.paused, "the pause gesture did not take effect")

        self._drive(app, len(released) * 8, start=len(paused_frames) / 30.0)
        self._drive(
            app,
            len(paused_frames),
            start=(len(paused_frames) + len(released) * 8) / 30.0,
        )
        app.shutdown()
        self.assertFalse(app.paused, "the pause could not be undone by gesture")

    def test_holding_the_pause_gesture_does_not_flip_flop(self) -> None:
        """Holding it must pause once, not toggle every time the cooldown expires.

        Resetting the debouncer on pause used to clear the action's cooldown, so
        the still-held gesture re-armed and resumed the application on its own.
        """
        app = self._app([hands([Finger.THUMB, Finger.PINKY], thumb_out=1.0)] * 90)
        self._drive(app, 90)
        app.shutdown()
        self.assertTrue(app.paused, "the paused state did not survive a sustained hold")

    def test_single_hand_pause_gesture_works_in_every_mode(self) -> None:
        """Each mode carries its own pause shape, so the escape is never missing."""
        for shape_name in ("shaka",):
            with self.subTest(shape=shape_name):
                from gesture_control.vocabulary import shape_table

                frames = []
                curls = up_shape(
                    [f for f in FINGERS if "up" in shape_table(shape_name).split(f.value + ":")[1][:4]]
                )
                del curls  # the helper below is clearer
                from gesture_control.gestures import parse_requirements, Requirement

                parsed = parse_requirements(shape_table(shape_name))
                curls = {
                    finger: (0.0 if parsed[finger] is Requirement.UP else 1.0)
                    for finger in FINGERS
                }
                thumb_out = 1.0 if parsed[Finger.THUMB] is Requirement.UP else 0.0
                frames = [((curls, thumb_out, Handedness.RIGHT),)] * 20

                config = harmless_config()
                from gesture_control.config import build_modes as _bm

                # Give the mode a single-hand pause gesture on the shaka shape.
                mode_entries = []
                for name in ("one", "two"):
                    mode_entries.append(
                        {
                            "name": name,
                            "label": name.upper(),
                            "gestures": [
                                {
                                    "name": "pause",
                                    "action": "pause_toggle",
                                    "requirements": SHAPES["shaka"],
                                    "label": "SHAKA",
                                }
                            ],
                        }
                    )
                config.modes = _bm(mode_entries)
                app = GestureApp(
                    config,
                    source=TestPatternSource(320, 180, fps=0.0),
                    detector=ScriptedHands(frames),
                    clock=virtual_clock(),
                )
                self._drive(app, 20)
                app.shutdown()
                self.assertTrue(app.paused)

    def test_paused_app_ignores_normal_gestures(self) -> None:
        """A pause must stop the controller acting on the machine."""
        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands([single([Finger.INDEX])] * 60),
            clock=virtual_clock(),
        )
        app.set_paused(True)
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        triggers = []
        for step in range(60):
            triggers.extend(app.process(frame, step / 30.0))
        app.shutdown()
        self.assertEqual(triggers, [], "a paused app fired a normal gesture")

    def test_pausing_resets_held_gestures(self) -> None:
        """Resuming must not immediately fire the gesture held before the pause."""
        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands([single([Finger.INDEX])] * 40),
            clock=virtual_clock(),
        )
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        for step in range(20):
            app.process(frame, step / 30.0)
        app.set_paused(True)
        app.set_paused(False)
        triggers = app.process(frame, 1.0)
        app.shutdown()
        self.assertEqual(triggers, [])

    def test_keyboard_p_toggles_pause(self) -> None:
        app = self._app([single([Finger.INDEX])])
        self.assertFalse(app.paused)
        self.assertTrue(app._handle_keys(ord("p")))
        self.assertTrue(app.paused)
        app._handle_keys(ord("p"))
        self.assertFalse(app.paused)
        app.shutdown()

    def test_keyboard_space_still_toggles_pause(self) -> None:
        """The original binding keeps working as well as ``p``."""
        app = self._app([single([Finger.INDEX])])
        app._handle_keys(ord(" "))
        self.assertTrue(app.paused)
        app.shutdown()

    def test_mode_keys_work_from_the_keyboard(self) -> None:
        app = self._app([single([Finger.INDEX])])
        app._handle_keys(ord("n"))
        self.assertEqual(app.mode_label, "TWO")
        app._handle_keys(ord("b"))
        self.assertEqual(app.mode_label, "ONE")
        app._handle_keys(ord("h"))
        self.assertEqual(app.mode_label, "ONE")
        app.shutdown()


class TestTwoHandQuit(unittest.TestCase):
    def test_two_hand_horns_quits(self) -> None:
        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=SyntheticHandDetector(
                GestureScript(QUIT_SCRIPT, rotation_span=0.0)
            ),
            clock=virtual_clock(),
        )
        summary = app.run(max_frames=120)
        self.assertLess(summary.frames, 120, "the quit gesture did not stop the loop")

    def test_one_hand_horns_does_not_quit(self) -> None:
        """One hand must never be enough to end the session."""
        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands(
                [single([Finger.INDEX, Finger.PINKY], thumb_out=0.0)] * 120
            ),
            clock=virtual_clock(),
        )
        summary = app.run(max_frames=60)
        self.assertEqual(summary.frames, 60, "the loop stopped early")

    def test_two_hand_gestures_need_a_longer_hold(self) -> None:
        """Quitting and mode changes require a deliberate hold, not a flash."""
        from gesture_control.config import default_config

        config = default_config("pro")
        self.assertGreater(
            config.tracking.two_hand_arm_seconds, config.tracking.arm_seconds
        )

    def test_a_brief_two_hand_flash_does_not_quit(self) -> None:
        config = harmless_config()
        # Three frames is ~0.1 s, far under the two-hand arming time.
        frames = [single([Finger.INDEX])] * 4 + [hands([Finger.INDEX, Finger.PINKY], thumb_out=0.0)] * 3
        frames += [single([Finger.INDEX])] * 30
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands(frames),
            clock=virtual_clock(),
        )
        summary = app.run(max_frames=len(frames))
        self.assertGreaterEqual(
            summary.frames, len(frames), "a brief flash of the quit shape ended the session"
        )


@unittest.skipUnless(HAVE_CV2, "OpenCV is required")
class TestModeHud(unittest.TestCase):
    def test_hud_shows_the_current_mode_and_its_gestures(self) -> None:
        from gesture_control.overlay import HudRenderer

        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands([single([Finger.INDEX])]),
            clock=virtual_clock(),
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = app.build_hud_state(0.0)
        app.shutdown()

        self.assertEqual(state.mode.label, "ONE")
        self.assertEqual(state.mode_count, 2)
        self.assertIsNotNone(state.gesture_set)
        self.assertEqual(state.gesture_set.names, ("g0", "g1"))

        rendered = HudRenderer(config.overlay).render(frame, state)
        self.assertGreater(int(rendered.sum()), 1000)

    def test_paused_hud_draws_the_resume_instructions(self) -> None:
        from gesture_control.overlay import HudRenderer

        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands([single([Finger.INDEX])]),
            clock=virtual_clock(),
        )
        app.set_paused(True)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        state = app.build_hud_state(0.0)
        app.shutdown()

        paused_frame = HudRenderer(config.overlay).render(frame.copy(), state)
        state.paused = False
        running_frame = HudRenderer(config.overlay).render(frame.copy(), state)
        self.assertGreater(
            int(paused_frame.sum()),
            int(running_frame.sum()),
            "the paused overlay did not add anything",
        )

    def test_two_hand_progress_is_shown(self) -> None:
        from gesture_control.overlay import HudRenderer

        config = harmless_config()
        app = GestureApp(
            config,
            source=TestPatternSource(320, 180, fps=0.0),
            detector=ScriptedHands([hands(FINGERS, thumb_out=1.0)] * 4),
            clock=virtual_clock(),
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for step in range(4):
            app.process(frame, step / 30.0)
        state = app.build_hud_state(0.2)
        app.shutdown()

        self.assertIsNotNone(state.two_hand_candidate)
        self.assertGreater(state.two_hand_armed_fraction, 0.0)
        rendered = HudRenderer(config.overlay).render(frame.copy(), state)
        self.assertEqual(rendered.shape, (720, 1280, 3))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
