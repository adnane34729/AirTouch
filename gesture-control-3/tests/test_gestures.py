"""Tests for gesture specs, ANY semantics, validation and the generated guide."""

from __future__ import annotations

import unittest

from gesture_control.config import PRESETS, build_gestures
from gesture_control.gestures import (
    GestureError,
    GestureSet,
    GestureSpec,
    Requirement,
    parse_requirements,
)
from gesture_control.types import FINGERS, Finger

from .helpers import default_gesture_set, fake_pose


class TestParseRequirements(unittest.TestCase):
    def test_string_form(self) -> None:
        parsed = parse_requirements("index:up,middle:down")
        self.assertIs(parsed[Finger.INDEX], Requirement.UP)
        self.assertIs(parsed[Finger.MIDDLE], Requirement.DOWN)
        self.assertIs(parsed[Finger.THUMB], Requirement.ANY)

    def test_mapping_form(self) -> None:
        parsed = parse_requirements({"index": "up", "pinky": "any"})
        self.assertIs(parsed[Finger.INDEX], Requirement.UP)
        self.assertIs(parsed[Finger.PINKY], Requirement.ANY)

    def test_already_parsed_form(self) -> None:
        parsed = parse_requirements({Finger.INDEX: Requirement.UP})
        self.assertIs(parsed[Finger.INDEX], Requirement.UP)

    def test_every_finger_present(self) -> None:
        self.assertEqual(set(parse_requirements("index:up")), set(FINGERS))

    def test_aliases(self) -> None:
        self.assertIs(parse_requirements("index:-")[Finger.INDEX], Requirement.ANY)
        self.assertIs(parse_requirements("index:ignore")[Finger.INDEX], Requirement.ANY)

    def test_unknown_finger_is_rejected(self) -> None:
        with self.assertRaises(GestureError):
            parse_requirements("thumbbb:up")

    def test_unknown_state_is_rejected(self) -> None:
        with self.assertRaises(GestureError):
            parse_requirements("index:sideways")

    def test_malformed_chunk_is_rejected(self) -> None:
        with self.assertRaises(GestureError):
            parse_requirements("index")


class TestAnySemantics(unittest.TestCase):
    """The core usability fix: a gesture constrains only the fingers it names."""

    def setUp(self) -> None:
        self.gestures = default_gesture_set()

    def _find(self, name: str) -> GestureSpec:
        spec = self.gestures.by_name(name)
        assert spec is not None
        return spec

    def test_thumb_is_ignored_by_finger_gestures(self) -> None:
        for name in ("left", "volume_up", "brightness_up", "volume_down"):
            with self.subTest(gesture=name):
                self.assertIs(
                    self._find(name).requirements[Finger.THUMB],
                    Requirement.ANY,
                    f"{name} should not constrain the thumb",
                )

    def test_thumb_position_does_not_break_index_gesture(self) -> None:
        """Both thumb positions must select the same gesture.

        The original required the thumb to be *down* for this gesture, so a hand
        with the thumb slightly out silently did nothing.
        """
        with_thumb_down = self.gestures.classify(fake_pose({Finger.INDEX}))
        with_thumb_up = self.gestures.classify(fake_pose({Finger.INDEX, Finger.THUMB}))
        self.assertIsNotNone(with_thumb_down)
        self.assertIsNotNone(with_thumb_up)
        self.assertEqual(with_thumb_down.name, "brightness_up")
        self.assertEqual(with_thumb_up.name, "brightness_up")

    def test_classic_fist_still_requires_the_thumb_down(self) -> None:
        """``space`` is the exception: a thumbs-up must not press space."""
        spec = self._find("space")
        self.assertIs(spec.requirements[Finger.THUMB], Requirement.DOWN)
        self.assertIsNone(
            self.gestures.classify(fake_pose({Finger.THUMB})),
            "a thumbs-up should not be mistaken for anything in the classic preset",
        )


class TestClassicMapping(unittest.TestCase):
    """The classic preset must reproduce the original's mapping exactly."""

    CASES = (
        (frozenset(), "space"),
        (frozenset({Finger.INDEX}), "brightness_up"),
        (frozenset({Finger.INDEX, Finger.MIDDLE}), "left"),
        (frozenset({Finger.PINKY}), "volume_down"),
        (frozenset({Finger.RING, Finger.PINKY}), "volume_up"),
    )

    def test_each_hand_shape_maps_to_the_original_action(self) -> None:
        gestures = default_gesture_set()
        expected_actions = {
            "space": "key_space",
            "brightness_up": "brightness_up",
            "left": "key_left",
            "volume_down": "volume_down",
            "volume_up": "volume_up",
        }
        for fingers, name in self.CASES:
            with self.subTest(gesture=name):
                match = gestures.classify(fake_pose(fingers))
                self.assertIsNotNone(match, f"no gesture matched {sorted(f.name for f in fingers)}")
                self.assertEqual(match.name, name)
                self.assertEqual(match.action, expected_actions[name])

    def test_open_hand_matches_nothing_in_classic(self) -> None:
        self.assertIsNone(default_gesture_set().classify(fake_pose(FINGERS)))

    def test_ring_alone_matches_nothing_in_classic(self) -> None:
        self.assertIsNone(default_gesture_set().classify(fake_pose({Finger.RING})))


class TestPresets(unittest.TestCase):
    def test_all_presets_build_and_validate(self) -> None:
        for name in PRESETS:
            with self.subTest(preset=name):
                GestureSet(specs=build_gestures(PRESETS[name].modes[0]["gestures"]))

    def test_no_preset_has_shadowing_warnings(self) -> None:
        for name in PRESETS:
            with self.subTest(preset=name):
                gesture_set = GestureSet(specs=build_gestures(PRESETS[name].modes[0]["gestures"]))
                self.assertEqual(gesture_set.warnings, (), f"{name}: {gesture_set.warnings}")

    def test_every_preset_gesture_has_a_distinct_action_or_shape(self) -> None:
        for name in PRESETS:
            gesture_set = GestureSet(specs=build_gestures(PRESETS[name].modes[0]["gestures"]))
            shapes = [spec.pose_text() for spec in gesture_set]
            self.assertEqual(len(shapes), len(set(shapes)), f"{name} has duplicate hand shapes")

    def test_media_preset_distinguishes_thumb_up_from_fist(self) -> None:
        gesture_set = GestureSet(specs=build_gestures(PRESETS["media"].modes[0]["gestures"]))
        thumb_up = gesture_set.classify(fake_pose({Finger.THUMB}))
        fist = gesture_set.classify(fake_pose(set()))
        self.assertIsNotNone(thumb_up)
        self.assertIsNotNone(fist)
        self.assertEqual(thumb_up.name, "play_pause")
        self.assertEqual(fist.name, "mute")
        self.assertNotEqual(thumb_up.name, fist.name)

    def test_media_preset_recognises_every_gesture_shape(self) -> None:
        gesture_set = GestureSet(specs=build_gestures(PRESETS["media"].modes[0]["gestures"]))
        for spec in gesture_set:
            with self.subTest(gesture=spec.name):
                extended = {
                    finger
                    for finger, requirement in spec.requirements.items()
                    if requirement is Requirement.UP
                }
                match = gesture_set.classify(fake_pose(extended))
                self.assertIsNotNone(match)
                self.assertEqual(match.name, spec.name)


class TestValidation(unittest.TestCase):
    def _spec(self, name: str, requirements: str) -> GestureSpec:
        return GestureSpec(name=name, action=name, requirements=parse_requirements(requirements))

    def test_identical_gestures_are_rejected(self) -> None:
        first = self._spec("a", "index:up,middle:down")
        second = self._spec("b", "index:up,middle:down")
        with self.assertRaises(GestureError) as ctx:
            GestureSet(specs=(first, second))
        self.assertIn("identical", str(ctx.exception))

    def test_duplicate_names_are_rejected(self) -> None:
        first = self._spec("a", "index:up")
        second = self._spec("a", "pinky:up")
        with self.assertRaises(GestureError) as ctx:
            GestureSet(specs=(first, second))
        self.assertIn("duplicate", str(ctx.exception))

    def test_unconstrained_gesture_is_rejected(self) -> None:
        with self.assertRaises(GestureError) as ctx:
            GestureSet(specs=(self._spec("anything", "index:any"),))
        self.assertIn("constrains no fingers", str(ctx.exception))

    def test_ambiguous_equal_specificity_is_rejected(self) -> None:
        """Two one-finger gestures that do not contradict can both match."""
        first = self._spec("a", "index:up")
        second = self._spec("b", "index:up,pinky:any")
        with self.assertRaises(GestureError):
            GestureSet(specs=(first, second))

    def test_subset_overlap_is_a_warning_not_an_error(self) -> None:
        general = self._spec("general", "index:up")
        specific = self._spec("specific", "index:up,middle:down,ring:down")
        gesture_set = GestureSet(specs=(general, specific))
        self.assertTrue(gesture_set.warnings, "expected a shadowing warning")
        self.assertIn("specific", gesture_set.warnings[0])

    def test_empty_set_is_rejected(self) -> None:
        with self.assertRaises(GestureError):
            GestureSet(specs=())

    def test_contradicting_gestures_are_fine(self) -> None:
        """Disjoint by construction -- no warning, no error."""
        up = self._spec("up", "index:up,middle:down")
        down = self._spec("down", "index:down,middle:up")
        gesture_set = GestureSet(specs=(up, down))
        self.assertEqual(gesture_set.warnings, ())


class TestSpecificity(unittest.TestCase):
    def test_more_specific_gesture_wins(self) -> None:
        general = GestureSpec(
            name="general", action="general", requirements=parse_requirements("index:up")
        )
        specific = GestureSpec(
            name="specific",
            action="specific",
            requirements=parse_requirements("index:up,middle:down"),
        )
        gesture_set = GestureSet(specs=(general, specific))
        match = gesture_set.classify(fake_pose({Finger.INDEX}))
        self.assertIsNotNone(match)
        self.assertEqual(match.name, "specific")


class TestPresentation(unittest.TestCase):
    def test_pose_text_is_thumb_first(self) -> None:
        """Unmentioned fingers render as "-" because they are genuinely ignored."""
        spec = GestureSpec(
            name="x", action="x", requirements=parse_requirements("index:up,pinky:up")
        )
        self.assertEqual(spec.pose_text(), "- ^ - - ^")

    def test_pose_text_for_a_fully_specified_gesture(self) -> None:
        spec = GestureSpec(
            name="x",
            action="x",
            requirements=parse_requirements("index:up,middle:down,ring:down,pinky:down,thumb:down"),
        )
        self.assertEqual(spec.pose_text(), "v ^ v v v")

    def test_requirement_text(self) -> None:
        spec = GestureSpec(
            name="x", action="x", requirements=parse_requirements("index:up,middle:down")
        )
        self.assertEqual(spec.requirement_text(), "index: up, middle: down")

    def test_guide_is_generated_from_specs(self) -> None:
        """The guide cannot drift from the recogniser: it is the same data."""
        gestures = default_gesture_set()
        lines = gestures.guide_lines()
        self.assertEqual(len(lines), len(gestures))
        for spec, line in zip(gestures, lines):
            self.assertIn(spec.pose_text(), line)
            self.assertIn(spec.label, line)

    def test_guide_covers_every_gesture(self) -> None:
        gestures = default_gesture_set()
        joined = "\n".join(gestures.guide_lines())
        for spec in gestures:
            self.assertIn(spec.label, joined)

    def test_guide_wrapping(self) -> None:
        gestures = default_gesture_set()
        wrapped = gestures.guide_lines(width=12)
        self.assertGreaterEqual(len(wrapped), len(gestures))

    def test_as_dict_round_trip(self) -> None:
        spec = GestureSpec(
            name="x",
            action="volume_up",
            requirements=parse_requirements("index:up"),
            label="INDEX",
            repeatable=True,
            cooldown=0.5,
        )
        payload = spec.as_dict()
        self.assertEqual(payload["action"], "volume_up")
        self.assertEqual(payload["cooldown"], 0.5)
        rebuilt = GestureSpec(
            name="x",
            action=payload["action"],
            requirements=parse_requirements(payload["requirements"]),
            label=payload["label"],
            repeatable=payload["repeatable"],
            cooldown=payload["cooldown"],
        )
        self.assertEqual(rebuilt.requirements, spec.requirements)

    def test_specificity_and_constraints(self) -> None:
        spec = GestureSpec(
            name="x", action="x", requirements=parse_requirements("index:up,middle:down")
        )
        self.assertEqual(spec.specificity, 2)
        self.assertTrue(spec.is_constrained)
        self.assertIn((Finger.INDEX, Requirement.UP), spec.constraints)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
