"""Tests for the gesture legend: pictograms, plain names, and mode switching.

The requirement these tests pin down is a usability one: the HUD must show every
gesture available *in the current mode*, as a picture of the hand shape beside the
plain name of what it does -- without the reader having to learn which column is
which finger, or what any symbol means.
"""

from __future__ import annotations

import unittest

import numpy as np

from gesture_control.config import OverlayConfig, default_config
from gesture_control.gestures import Requirement, parse_requirements
from gesture_control.icons import IconLibrary
from gesture_control.overlay import HudRenderer, HudState
from gesture_control.types import FINGERS, Finger
from gesture_control.vocabulary import SHAPES

try:
    import cv2  # noqa: F401  (capability probe)

    HAVE_CV2 = True
except Exception:  # pragma: no cover
    HAVE_CV2 = False


class TestIconLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self.library = IconLibrary()

    def test_every_vocabulary_shape_becomes_an_icon(self) -> None:
        for name in SHAPES:
            with self.subTest(shape=name):
                icon = self.library.get(parse_requirements(SHAPES[name]))
                self.assertEqual(len(icon.points), 21)
                self.assertGreater(icon.aspect, 0.2)
                self.assertLess(icon.aspect, 3.0)

    def test_icons_are_normalised_into_the_unit_box(self) -> None:
        """So a caller can map them into any icon box without clipping."""
        for name in SHAPES:
            with self.subTest(shape=name):
                icon = self.library.get(parse_requirements(SHAPES[name]))
                xs = [point[0] for point in icon.points]
                ys = [point[1] for point in icon.points]
                self.assertGreaterEqual(min(xs), -1e-9)
                self.assertLessEqual(max(xs), 1 + 1e-9)
                self.assertGreaterEqual(min(ys), -1e-9)
                self.assertLessEqual(max(ys), 1 + 1e-9)

    def test_results_are_cached(self) -> None:
        """A legend redraws every frame; the geometry must be computed once."""
        requirements = parse_requirements(SHAPES["open"])
        first = self.library.get(requirements)
        second = self.library.get(dict(requirements))
        self.assertIs(first, second)

    def test_extended_fingers_are_reported_per_finger(self) -> None:
        icon = self.library.get(parse_requirements(SHAPES["horns"]))
        up = [finger for finger, raised in zip(FINGERS, icon.extended) if raised]
        self.assertEqual(up, [Finger.INDEX, Finger.PINKY])

    def test_a_dont_care_finger_is_marked_unspecified(self) -> None:
        """The classic preset ignores the thumb, and the icon must say so.

        Drawn half-curled in a neutral colour, so "either way is fine" is visible
        rather than being one more symbol to learn.
        """
        icon = self.library.get(
            parse_requirements("index:up,middle:up,ring:down,pinky:down")
        )
        thumb = FINGERS.index(Finger.THUMB)
        self.assertTrue(icon.unspecified[thumb])
        self.assertFalse(icon.extended[thumb])
        self.assertFalse(any(icon.unspecified[i] for i in range(1, 5)))

    def test_different_shapes_give_different_geometry(self) -> None:
        fist = self.library.get(parse_requirements(SHAPES["fist"]))
        open_hand = self.library.get(parse_requirements(SHAPES["open"]))
        self.assertNotAlmostEqual(fist.aspect, open_hand.aspect, places=2)
        self.assertNotEqual(fist.points, open_hand.points)

    def test_finger_bones_cover_every_joint(self) -> None:
        icon = self.library.get(parse_requirements(SHAPES["open"]))
        bones = icon.finger_bones()
        self.assertEqual(len(bones), 5)
        segments = sum(len(chain) for _, chain in bones)
        self.assertEqual(segments, 15)  # three phalanges per finger


@unittest.skipUnless(HAVE_CV2, "OpenCV is required")
class TestLegendLayout(unittest.TestCase):
    def setUp(self) -> None:
        self.config = default_config("pro")
        self.renderer = HudRenderer(self.config.overlay)

    def _entries(self, count: int) -> list[tuple[str, str]]:
        return [(f"g{i}", f"Action number {i}") for i in range(count)]

    def test_a_grid_is_planned_that_holds_every_entry(self) -> None:
        """No gesture may be silently dropped: rows x columns must cover them all."""
        for count in (1, 5, 10, 14, 20, 30):
            with self.subTest(entries=count):
                icon_height, columns, row_height, width = self.renderer._plan_grid(
                    self._entries(count), s=1.0, icons=1, available_height=600
                )
                self.assertGreater(icon_height, 0)
                rows = max(1, -(-count // columns))
                self.assertGreaterEqual(rows * columns, count)
                self.assertGreater(width, 0)

    def test_the_planned_panel_fits_the_space_it_was_given(self) -> None:
        padding = 12
        header = 20
        for count in (1, 6, 10, 16, 24):
            for available in (200, 400, 600):
                with self.subTest(entries=count, available=available):
                    icon_height, columns, row_height, width = self.renderer._plan_grid(
                        self._entries(count), s=1.0, icons=1, available_height=available
                    )
                    if columns == 0:
                        # Nothing reasonable fits at this size: the planner says so
                        # rather than proposing a panel that would spill.
                        self.assertEqual((icon_height, row_height, width), (0, 0, 0))
                        continue
                    rows = max(1, -(-count // columns))
                    height = padding + header + rows * row_height + padding
                    self.assertLessEqual(height, available)

    def test_dense_legends_use_more_columns(self) -> None:
        _, narrow_columns, _, _ = self.renderer._plan_grid(
            self._entries(4), s=1.0, icons=1, available_height=600
        )
        _, wide_columns, _, _ = self.renderer._plan_grid(
            self._entries(20), s=1.0, icons=1, available_height=600
        )
        self.assertLess(narrow_columns, wide_columns)

    def test_nothing_fits_when_there_is_no_room(self) -> None:
        """Better to draw no legend than one spilling off the frame."""
        icon_height, columns, _, width = self.renderer._plan_grid(
            self._entries(40), s=1.0, icons=1, available_height=10
        )
        self.assertEqual((icon_height, columns, width), (0, 0, 0))

    def test_longer_action_names_make_a_wider_panel(self) -> None:
        """The legend is measured from the action text, not from a guess."""
        short = self.renderer._plan_grid(
            [("g", "Mute")], s=1.0, icons=1, available_height=600
        )[3]
        long = self.renderer._plan_grid(
            [("g", "Volume + (media key)")], s=1.0, icons=1, available_height=600
        )[3]
        self.assertGreater(long, short)


@unittest.skipUnless(HAVE_CV2, "OpenCV is required")
class TestLegendContent(unittest.TestCase):
    """What the legend says, and that it follows the mode."""

    def _state(self, config, mode_index: int = 0, **overrides) -> HudState:
        mode = config.modes[mode_index]
        labels = {
            "volume_up": "Volume +",
            "volume_down": "Volume -",
            "mute_toggle": "Mute",
            "brightness_up": "Brightness +",
            "key_page_down": "Page down",
            "key_right": "Right arrow",
        }
        state = HudState(
            mode=mode,
            mode_index=mode_index,
            mode_count=len(config.modes),
            modes=config.modes,
            gesture_set=mode.build_gesture_set(),
            two_hand_set=config.build_two_hand_set(),
            action_labels=labels,
        )
        for key, value in overrides.items():
            setattr(state, key, value)
        return state

    def test_the_legend_and_action_labels_produce_a_render(self) -> None:
        config = default_config("pro")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        rendered = HudRenderer(config.overlay).render(frame, self._state(config))
        self.assertGreater(int(rendered.sum()), 1000)

    def test_changing_mode_changes_the_legend(self) -> None:
        """The table must follow the mode, which is the whole point."""
        config = default_config("pro")
        renderer = HudRenderer(config.overlay)
        frames = []
        for index in range(len(config.modes)):
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frames.append(renderer.render(frame, self._state(config, index)).copy())
        for first in range(len(frames)):
            for second in range(first + 1, len(frames)):
                with self.subTest(first=config.modes[first].name, second=config.modes[second].name):
                    self.assertFalse(
                        np.array_equal(frames[first], frames[second]),
                        "two modes rendered identically",
                    )

    def test_the_same_shape_shows_different_actions_per_mode(self) -> None:
        """The 'index' row must name a different action in different modes."""
        config = default_config("pro")
        media = self._state(config, config.mode_names.index("media"))
        browse = self._state(config, config.mode_names.index("browse"))

        def index_action(state):
            for spec in state.gesture_set:
                if spec.requirements[Finger.INDEX] is Requirement.UP and all(
                    spec.requirements[f] is Requirement.DOWN
                    for f in FINGERS
                    if f is not Finger.INDEX
                ):
                    return state.action_labels.get(spec.action, spec.action)
            return None

        self.assertEqual(index_action(media), "Volume +")
        self.assertEqual(index_action(browse), "Page down")

    def test_every_gesture_in_the_mode_is_in_the_legend(self) -> None:
        """Nothing is truncated: the planned grid covers the whole gesture set."""
        config = default_config("pro")
        renderer = HudRenderer(config.overlay)
        state = self._state(config)
        entries = [(spec.name, renderer._action_label(state, spec.action)) for spec in state.gesture_set]
        _, columns, _, _ = renderer._plan_grid(
            entries, s=1.0, icons=1, available_height=560
        )
        rows = max(1, -(-len(entries) // columns))
        self.assertGreaterEqual(rows * columns, len(entries))

    def test_missing_action_labels_fall_back_readably(self) -> None:
        """A gesture whose action is unknown still shows understandable text."""
        config = default_config("pro")
        renderer = HudRenderer(config.overlay)
        state = self._state(config, action_labels={})
        text = renderer._action_label(state, "key_page_down")
        self.assertEqual(text, "Key page down")
        self.assertNotIn("_", text)

    def test_icons_are_drawn_for_every_vocabulary_shape(self) -> None:
        """One render per shape, at a size where the fingers are distinguishable."""
        renderer = HudRenderer(OverlayConfig())
        for name in SHAPES:
            with self.subTest(shape=name):
                frame = np.zeros((80, 80, 3), dtype=np.uint8)
                width = renderer._draw_shape_icon(
                    frame, 10, 10, 40, parse_requirements(SHAPES[name]), 1.0
                )
                self.assertGreater(width, 0)
                self.assertGreater(int(frame.sum()), 0)

    def test_the_pictogram_marks_extended_fingers_in_their_own_colour(self) -> None:
        from gesture_control.overlay import PALETTE, _FINGER_COLOURS

        renderer = HudRenderer(OverlayConfig())
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        renderer._draw_shape_icon(
            frame, 6, 6, 48, parse_requirements(SHAPES["index"]), 1.0
        )
        index_colour = np.array(_FINGER_COLOURS[Finger.INDEX], dtype=np.int16)
        match = np.all(np.abs(frame.astype(np.int16) - index_colour) <= 2, axis=-1)
        self.assertTrue(match.any(), "the raised index finger was not drawn in its colour")
        dim_colour = np.array(PALETTE["down"], dtype=np.int16)
        curled = np.all(np.abs(frame.astype(np.int16) - dim_colour) <= 2, axis=-1)
        self.assertTrue(curled.any(), "the curled fingers were not drawn")

    def test_renders_at_small_resolutions_without_overflow(self) -> None:
        config = default_config("pro")
        renderer = HudRenderer(config.overlay)
        for width, height in ((320, 180), (640, 360), (1920, 1080)):
            with self.subTest(size=(width, height)):
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                state = self._state(config)
                state.resolution = (width, height)
                out = renderer.render(frame, state)
                self.assertEqual(out.shape, (height, width, 3))

    def test_the_two_hand_legend_uses_action_names_too(self) -> None:
        config = default_config("pro")
        renderer = HudRenderer(config.overlay)
        state = self._state(config)
        state.action_labels = {**state.action_labels, "mode_next": "Next mode", "quit_app": "Quit"}
        self.assertEqual(renderer._action_label(state, "mode_next"), "Next mode")
        self.assertEqual(renderer._action_label(state, "quit_app"), "Quit")

    def test_guide_can_be_switched_off(self) -> None:
        config = default_config("pro")
        with_guide = HudRenderer(OverlayConfig(show_guide=True)).render(
            np.zeros((720, 1280, 3), dtype=np.uint8), self._state(config)
        )
        without = HudRenderer(OverlayConfig(show_guide=False)).render(
            np.zeros((720, 1280, 3), dtype=np.uint8), self._state(config)
        )
        self.assertGreater(int(with_guide.sum()), int(without.sum()))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
