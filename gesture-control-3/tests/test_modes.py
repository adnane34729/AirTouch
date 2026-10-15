"""Tests for modes and two-hand gestures.

Two-hand gestures are the gate on the destructive and disorienting actions, so
the property that matters most is asserted explicitly and repeatedly: **one hand
is never enough.**
"""

from __future__ import annotations

import unittest

from gesture_control.gestures import GestureSpec, parse_requirements
from gesture_control.modes import (
    Mode,
    ModeError,
    ModeSet,
    TwoHandGesture,
    TwoHandGestureSet,
    shape,
)
from gesture_control.types import FINGERS, Finger

from .helpers import fake_pose, mode_gesture_set, pro_config, real_pose, two_hand_set


def spec(name: str, action: str, requirements: str, **kwargs) -> GestureSpec:
    return GestureSpec(
        name=name, action=action, requirements=parse_requirements(requirements), **kwargs
    )


OPEN = "thumb:up,index:up,middle:up,ring:up,pinky:up"
FIST = "thumb:down,index:down,middle:down,ring:down,pinky:down"
INDEX = "thumb:down,index:up,middle:down,ring:down,pinky:down"
PINKY = "thumb:down,index:down,middle:down,ring:down,pinky:up"
PEACE = "thumb:down,index:up,middle:up,ring:down,pinky:down"
SHAKA = "thumb:up,index:down,middle:down,ring:down,pinky:up"


class TestMode(unittest.TestCase):
    def test_label_defaults_to_the_uppercased_name(self) -> None:
        self.assertEqual(Mode(name="media").label, "MEDIA")

    def test_empty_mode_is_legal_and_classifies_nothing(self) -> None:
        """A deliberate "rest your hand" mode: valid, but inert."""
        mode = Mode(name="safe", label="SAFE")
        self.assertTrue(mode.is_empty)
        self.assertIsNone(mode.build_gesture_set())
        self.assertIsNone(mode.classify(fake_pose({Finger.INDEX})))

    def test_mode_without_a_name_is_rejected(self) -> None:
        with self.assertRaises(ModeError):
            Mode(name="")

    def test_conflicting_gestures_in_a_mode_are_rejected_at_build(self) -> None:
        """Ambiguity inside a mode must fail loudly, not resolve by luck."""
        mode = Mode(
            name="bad",
            gestures=(spec("a", "volume_up", "index:up"), spec("b", "volume_down", "index:up")),
        )
        with self.assertRaises(Exception):
            mode.build_gesture_set()

    def test_fully_specified_shapes_can_never_be_ambiguous(self) -> None:
        """Complete shapes always disagree somewhere, so no two can both match."""
        shapes = [OPEN, FIST, INDEX, PINKY, PEACE, SHAKA]
        mode = Mode(
            name="many",
            gestures=tuple(
                spec(f"g{i}", "noop", s) for i, s in enumerate(shapes)
            ),
        )
        gesture_set = mode.build_gesture_set()
        self.assertIsNotNone(gesture_set)
        self.assertEqual(gesture_set.warnings, ())

    def test_iteration_and_length(self) -> None:
        mode = Mode(name="m", gestures=(spec("a", "noop", INDEX), spec("b", "noop", PINKY)))
        self.assertEqual(len(mode), 2)
        self.assertEqual([s.name for s in mode], ["a", "b"])


class TestModeSet(unittest.TestCase):
    def _modes(self, count: int = 3) -> ModeSet:
        return ModeSet(
            modes=tuple(
                Mode(name=f"m{i}", label=f"M{i}", gestures=(spec("g", "noop", INDEX),))
                for i in range(count)
            )
        )

    def test_basic_container_behaviour(self) -> None:
        mode_set = self._modes(3)
        self.assertEqual(len(mode_set), 3)
        self.assertEqual(mode_set.names, ("m0", "m1", "m2"))
        self.assertEqual(mode_set[1].name, "m1")

    def test_empty_mode_set_is_rejected(self) -> None:
        with self.assertRaises(ModeError):
            ModeSet(modes=())

    def test_duplicate_mode_names_are_rejected(self) -> None:
        with self.assertRaises(ModeError):
            ModeSet(modes=(Mode(name="a"), Mode(name="a")))

    def test_index_of_is_case_insensitive(self) -> None:
        self.assertEqual(self._modes().index_of("M1"), 1)

    def test_index_of_unknown_mode_lists_the_options(self) -> None:
        with self.assertRaises(ModeError) as ctx:
            self._modes().index_of("nope")
        self.assertIn("m0", str(ctx.exception))

    def test_step_wraps_by_default(self) -> None:
        mode_set = self._modes(3)
        self.assertEqual(mode_set.step(2, 1), 0)
        self.assertEqual(mode_set.step(0, -1), 2)

    def test_step_clamps_when_wrapping_is_off(self) -> None:
        mode_set = ModeSet(
            modes=tuple(Mode(name=f"m{i}") for i in range(3)), wrap=False
        )
        self.assertEqual(mode_set.step(2, 1), 2)
        self.assertEqual(mode_set.step(0, -1), 0)

    def test_single_mode_flag(self) -> None:
        self.assertTrue(self._modes(1).single_mode)
        self.assertFalse(self._modes(2).single_mode)


class TestTwoHandGesture(unittest.TestCase):
    def test_symmetric_matches_two_identical_hands(self) -> None:
        gesture = TwoHandGesture(name="next", action="mode_next", shape=shape(OPEN))
        self.assertTrue(gesture.matches(fake_pose(FINGERS), fake_pose(FINGERS)))

    def test_symmetric_rejects_a_mismatched_pair(self) -> None:
        gesture = TwoHandGesture(name="next", action="mode_next", shape=shape(OPEN))
        self.assertFalse(gesture.matches(fake_pose(FINGERS), fake_pose({Finger.INDEX})))

    def test_symmetric_ignores_which_hand_is_which(self) -> None:
        """A wrong handedness label must not silently disable a mode change."""
        gesture = TwoHandGesture(name="next", action="mode_next", shape=shape(OPEN))
        self.assertTrue(gesture.matches(fake_pose(FINGERS), fake_pose(FINGERS)))

    def test_ordered_matches_either_assignment(self) -> None:
        gesture = TwoHandGesture(
            name="combo", action="noop", left=shape(OPEN), right=shape(FIST)
        )
        self.assertTrue(gesture.matches(fake_pose(FINGERS), fake_pose(set())))
        # Swapped, as a mislabelled detection would present it.
        self.assertTrue(gesture.matches(fake_pose(set()), fake_pose(FINGERS)))

    def test_ordered_rejects_the_wrong_pair(self) -> None:
        gesture = TwoHandGesture(
            name="combo", action="noop", left=shape(OPEN), right=shape(FIST)
        )
        self.assertFalse(gesture.matches(fake_pose({Finger.INDEX}), fake_pose(set())))

    def test_cannot_be_both_symmetric_and_ordered(self) -> None:
        with self.assertRaises(ModeError):
            TwoHandGesture(
                name="x", action="noop", shape=shape(OPEN), left=shape(FIST)
            )

    def test_needs_some_shape(self) -> None:
        with self.assertRaises(ModeError):
            TwoHandGesture(name="x", action="noop")

    def test_unconstrained_gesture_is_rejected(self) -> None:
        with self.assertRaises(ModeError):
            TwoHandGesture(name="x", action="noop", shape=parse_requirements("index:any"))

    def test_hand_text_renders_both_hands(self) -> None:
        gesture = TwoHandGesture(name="x", action="noop", shape=shape(INDEX))
        self.assertIn("+", gesture.hand_text())
        self.assertEqual(gesture.hand_text().count("^"), 2)

    def test_as_dict_round_trips(self) -> None:
        gesture = TwoHandGesture(
            name="x", action="noop", left=shape(OPEN), right=shape(FIST), cooldown=0.5
        )
        payload = gesture.as_dict()
        self.assertIn("left", payload)
        self.assertIn("right", payload)
        self.assertEqual(payload["cooldown"], 0.5)

    def test_constraints_counts_the_declared_requirements(self) -> None:
        """Specificity, so the more detailed gesture wins a tie.

        A symmetric gesture declares one table (applied to both hands), an
        ordered one declares two -- and the count reflects what was written, so
        two gestures can be compared like for like.
        """
        symmetric = TwoHandGesture(name="x", action="noop", shape=shape(OPEN))
        self.assertEqual(symmetric.constraints, 5)
        ordered = TwoHandGesture(
            name="y", action="noop", left=shape(OPEN), right=shape(INDEX)
        )
        self.assertEqual(ordered.constraints, 10)


class TestTwoHandGestureSet(unittest.TestCase):
    def test_one_hand_is_never_enough(self) -> None:
        """The safety property everything else depends on."""
        gesture_set = TwoHandGestureSet(
            gestures=(TwoHandGesture(name="quit", action="quit_app", shape=shape(SHAKA)),)
        )
        one_hand = real_pose({Finger.THUMB, Finger.PINKY}, thumb_out=1.0)
        self.assertIsNone(gesture_set.classify(one_hand, None))
        self.assertIsNone(gesture_set.classify(None, one_hand))
        self.assertIsNone(gesture_set.classify(None, None))

    def test_classifies_a_matching_pair(self) -> None:
        gesture_set = TwoHandGestureSet(
            gestures=(TwoHandGesture(name="quit", action="quit_app", shape=shape(SHAKA)),)
        )
        left = real_pose({Finger.THUMB, Finger.PINKY}, thumb_out=1.0)
        right = real_pose({Finger.THUMB, Finger.PINKY}, thumb_out=1.0)
        match = gesture_set.classify(left, right)
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "quit")

    def test_duplicate_names_rejected(self) -> None:
        one = TwoHandGesture(name="a", action="noop", shape=shape(OPEN))
        two = TwoHandGesture(name="a", action="noop", shape=shape(FIST))
        with self.assertRaises(ModeError):
            TwoHandGestureSet(gestures=(one, two))

    def test_contradicting_gestures_are_disjoint(self) -> None:
        gesture_set = TwoHandGestureSet(
            gestures=(
                TwoHandGesture(name="a", action="noop", shape=shape(OPEN)),
                TwoHandGesture(name="b", action="noop", shape=shape(FIST)),
            )
        )
        self.assertEqual(gesture_set.warnings, ())

    def test_overlapping_gestures_are_reported(self) -> None:
        """A less specific gesture that cannot contradict a broader one is shadowed.

        Reported rather than rejected, because the specific one wins
        deterministically -- but it is almost never what the author intended.
        """
        gesture_set = TwoHandGestureSet(
            gestures=(
                TwoHandGesture(name="broad", action="noop", shape=parse_requirements("index:up")),
                TwoHandGesture(name="narrow", action="noop", shape=shape(INDEX)),
            )
        )
        self.assertTrue(gesture_set.warnings, "expected a shadowing warning")
        self.assertIn("narrow", gesture_set.warnings[0])

    def test_identical_gestures_are_rejected(self) -> None:
        with self.assertRaises(ModeError):
            TwoHandGestureSet(
                gestures=(
                    TwoHandGesture(name="a", action="noop", shape=shape(INDEX)),
                    TwoHandGesture(name="b", action="noop", shape=shape(INDEX)),
                )
            )

    def test_by_name_and_iteration(self) -> None:
        gesture_set = TwoHandGestureSet(
            gestures=(TwoHandGesture(name="a", action="noop", shape=shape(OPEN)),)
        )
        self.assertIsNotNone(gesture_set.by_name("a"))
        self.assertIsNone(gesture_set.by_name("zzz"))
        self.assertEqual(len(gesture_set), 1)
        self.assertEqual(gesture_set.names, ("a",))
        self.assertEqual(len(gesture_set.guide_lines()), 1)


class TestProPresetModes(unittest.TestCase):
    def test_every_mode_has_a_gesture_set(self) -> None:
        config = pro_config()
        for mode in config.modes:
            with self.subTest(mode=mode.name):
                self.assertGreater(len(mode), 0)
                self.assertIsNotNone(mode.build_gesture_set())

    def test_every_mode_can_pause(self) -> None:
        """A pause must be reversible from any mode, so every mode has the gesture."""
        config = pro_config()
        for mode in config.modes:
            with self.subTest(mode=mode.name):
                actions = [spec.action for spec in mode]
                self.assertIn("pause_toggle", actions)

    def test_the_same_shape_means_different_things_per_mode(self) -> None:
        """The whole point of modes, asserted directly."""
        config = pro_config()
        by_mode = {
            mode.name: {
                spec.pose_text(): spec.action for spec in mode
            }
            for mode in config.modes
        }
        index_shape = by_mode["media"]
        self.assertIn("v ^ v v v", index_shape)
        self.assertEqual(index_shape["v ^ v v v"], "volume_up")
        self.assertEqual(by_mode["system"]["v ^ v v v"], "brightness_up")
        self.assertEqual(by_mode["browse"]["v ^ v v v"], "key_page_down")
        self.assertEqual(by_mode["present"]["v ^ v v v"], "key_right")

    def test_no_mode_shadows_its_own_gestures(self) -> None:
        for mode in pro_config().modes:
            with self.subTest(mode=mode.name):
                self.assertEqual(mode.build_gesture_set().warnings, ())

    def test_start_mode_resolves(self) -> None:
        config = pro_config()
        self.assertEqual(config.switching.start, "media")
        self.assertEqual(config.switching.start_index(config.mode_names), 0)

    def test_two_hand_set_covers_switching_pause_and_quit(self) -> None:
        actions = {g.action for g in two_hand_set()}
        self.assertIn("mode_next", actions)
        self.assertIn("mode_prev", actions)
        self.assertIn("mode_home", actions)
        self.assertIn("pause_toggle", actions)
        self.assertIn("quit_app", actions)

    def test_one_hand_preset_still_gets_pause_and_quit(self) -> None:
        """Escape hatches must not depend on the preset that is loaded."""
        for preset in ("classic", "media", "minimal", "present"):
            with self.subTest(preset=preset):
                from gesture_control.config import default_config

                actions = {g.action for g in default_config(preset).build_two_hand_set()}
                self.assertIn("pause_toggle", actions)
                self.assertIn("quit_app", actions)

    def test_media_mode_gestures_classify_correctly(self) -> None:
        """Shape-to-action for one mode, through the real classifier."""
        gesture_set = mode_gesture_set("pro", "media")
        cases = (
            (FINGERS, "play_pause"),
            (set(), "mute"),
            ({Finger.INDEX}, "volume_up"),
            ({Finger.INDEX, Finger.MIDDLE}, "volume_down"),
            ({Finger.PINKY}, "next"),
            ({Finger.RING, Finger.PINKY}, "prev"),
        )
        for up, expected in cases:
            with self.subTest(up=sorted(f.value for f in up)):
                match = gesture_set.classify(fake_pose(up))
                self.assertIsNotNone(match, f"{sorted(f.value for f in up)} matched nothing")
                self.assertEqual(match.name, expected)

    def test_system_mode_gestures_classify_correctly(self) -> None:
        gesture_set = mode_gesture_set("pro", "system")
        cases = (
            ({Finger.INDEX}, "brightness_up"),
            ({Finger.INDEX, Finger.MIDDLE}, "brightness_down"),
            ({Finger.PINKY}, "volume_down"),
            ({Finger.RING, Finger.PINKY}, "volume_up"),
            (set(), "mute"),
            (FINGERS, "show_desktop"),
        )
        for up, expected in cases:
            with self.subTest(up=sorted(f.value for f in up)):
                match = gesture_set.classify(fake_pose(up))
                self.assertIsNotNone(match)
                self.assertEqual(match.name, expected)


class TestShapeVocabulary(unittest.TestCase):
    def test_all_shapes_parse(self) -> None:
        from gesture_control.vocabulary import SHAPES, shape_table

        for name in SHAPES:
            with self.subTest(shape=name):
                parsed = parse_requirements(shape_table(name))
                self.assertEqual(set(parsed), set(FINGERS))

    def test_shapes_are_fully_specified(self) -> None:
        """Complete shapes are what makes ambiguity structurally impossible."""
        from gesture_control.gestures import Requirement
        from gesture_control.vocabulary import SHAPES, shape_table

        for name in SHAPES:
            with self.subTest(shape=name):
                parsed = parse_requirements(shape_table(name))
                unspecified = [f.value for f in FINGERS if parsed[f] is Requirement.ANY]
                self.assertEqual(unspecified, [], f"shape {name!r} leaves fingers unconstrained")

    def test_shapes_are_pairwise_distinct(self) -> None:
        from gesture_control.vocabulary import SHAPES, shape_table

        seen: dict[str, str] = {}
        for name in SHAPES:
            table = shape_table(name)
            self.assertNotIn(table, seen, f"{name!r} duplicates {seen.get(table)!r}")
            seen[table] = name

    def test_unknown_shape_lists_the_known_ones(self) -> None:
        from gesture_control.vocabulary import shape_table

        with self.assertRaises(KeyError) as ctx:
            shape_table("nonsense")
        self.assertIn("known shapes", str(ctx.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
