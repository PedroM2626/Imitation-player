"""
Action emission: turn a binary action vector into real input events.

The old environment hard-coded index -> button for gamepad mode and ignored the
profile's ``mappings`` table entirely, so the table was decorative and the two
input modes behaved differently. Here both modes are driven by the same table,
which is validated against the model's action space at construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

XBOX_BUTTONS = {
    "A": "XUSB_GAMEPAD_A",
    "B": "XUSB_GAMEPAD_B",
    "X": "XUSB_GAMEPAD_X",
    "Y": "XUSB_GAMEPAD_Y",
    "LB": "XUSB_GAMEPAD_LEFT_SHOULDER",
    "RB": "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "L3": "XUSB_GAMEPAD_LEFT_THUMB",
    "R3": "XUSB_GAMEPAD_RIGHT_THUMB",
    "BACK": "XUSB_GAMEPAD_BACK",
    "START": "XUSB_GAMEPAD_START",
    "DPAD_UP": "XUSB_GAMEPAD_DPAD_UP",
    "DPAD_DOWN": "XUSB_GAMEPAD_DPAD_DOWN",
    "DPAD_LEFT": "XUSB_GAMEPAD_DPAD_LEFT",
    "DPAD_RIGHT": "XUSB_GAMEPAD_DPAD_RIGHT",
    "GUIDE": "XUSB_GAMEPAD_GUIDE",
}


def validate_mappings(mappings: Sequence[dict[str, Any]], num_actions: int) -> None:
    """Fail early if the table and the declared action width disagree."""
    if len(mappings) != num_actions:
        raise ValueError(
            f"actions.mappings has {len(mappings)} entries but actions.num_actions is "
            f"{num_actions}; they must be equal"
        )
    for i, m in enumerate(mappings):
        kind = m.get("kind")
        if kind == "button" and m.get("button") not in XBOX_BUTTONS and m.get("key") is None:
            raise ValueError(f"mapping[{i}] {m.get('name')!r}: unknown button {m.get('button')!r}")
        if kind == "axis" and (
            m.get("stick") not in ("left", "right") or m.get("axis") not in ("x", "y")
        ):
            raise ValueError(f"mapping[{i}] {m.get('name')!r}: axis needs stick and x|y")
        if kind == "trigger" and m.get("trigger") not in ("left", "right"):
            raise ValueError(f"mapping[{i}] {m.get('name')!r}: trigger needs left|right")
        if kind not in ("button", "axis", "trigger", "key", "mouse_button"):
            raise ValueError(f"mapping[{i}] {m.get('name')!r}: unknown kind {kind!r}")


class GamepadEmitter:
    """Drives a virtual Xbox 360 pad from the profile's mapping table."""

    def __init__(self, mappings: list[dict[str, Any]], vg_module) -> None:
        validate_mappings(mappings, len(mappings))
        self.mappings = mappings
        self._buttons = {
            m["button"]: getattr(vg_module.XUSB_BUTTON, XBOX_BUTTONS[m["button"]])
            for m in mappings
            if m.get("kind") == "button"
        }
        self.gamepad = vg_module.VX360Gamepad()
        self._held: set = set()
        self._axes: dict[str, float] = {
            "left_x": 0.0,
            "left_y": 0.0,
            "right_x": 0.0,
            "right_y": 0.0,
        }
        self._triggers: dict[str, int] = {"left": 0, "right": 0}

    def _resolve(self, active: np.ndarray) -> None:
        pressed = set()
        axes = {"left_x": 0.0, "left_y": 0.0, "right_x": 0.0, "right_y": 0.0}
        triggers = {"left": 0, "right": 0}

        for i, m in enumerate(self.mappings):
            if not active[i]:
                continue
            kind = m["kind"]
            if kind == "button":
                pressed.add(m["button"])
            elif kind == "axis":
                key = f"{m['stick']}_{m['axis']}"
                # Conflicting flags on one axis cancel out rather than last-write-wins.
                axes[key] = max(-1.0, min(1.0, axes[key] + float(m["value"])))
            elif kind == "trigger":
                triggers[m["trigger"]] = int(m.get("value", 255))

        self._pressed = pressed
        self._axes_next = axes
        self._triggers_next = triggers

    def apply(self, active: np.ndarray) -> None:
        self._resolve(active)

        for name in self._pressed - self._held:
            self.gamepad.press_button(self._buttons[name])
        for name in self._held - self._pressed:
            self.gamepad.release_button(self._buttons[name])
        self._held = self._pressed

        self._axes = self._axes_next
        self.gamepad.left_joystick_float(
            x_value_float=self._axes["left_x"], y_value_float=self._axes["left_y"]
        )
        self.gamepad.right_joystick_float(
            x_value_float=self._axes["right_x"], y_value_float=self._axes["right_y"]
        )

        for side in ("left", "right"):
            value = self._triggers_next[side]
            if side == "left":
                self.gamepad.left_trigger(value=value)
            else:
                self.gamepad.right_trigger(value=value)
            self._triggers[side] = value

        self.gamepad.update()

    def release_all(self) -> None:
        self.apply(np.zeros(len(self.mappings), dtype=np.float32))

    def close(self) -> None:
        self.release_all()


class KeyboardMouseEmitter:
    """Synthesises key and mouse events through pydirectinput."""

    def __init__(self, mappings: list[dict[str, Any]]) -> None:
        validate_mappings(mappings, len(mappings))
        import pydirectinput

        pydirectinput.FAILSAFE = False
        self._p = pydirectinput
        self.mappings = mappings
        self._held: set = set()

    def apply(self, active: np.ndarray) -> None:
        current = {i for i, v in enumerate(active) if v}
        for i in current - self._held:
            self._press(self.mappings[i], True)
        for i in self._held - current:
            self._press(self.mappings[i], False)
        self._held = current

    def _press(self, m: dict[str, Any], down: bool) -> None:
        if m["kind"] == "key":
            getattr(self._p, "keyDown" if down else "keyUp")(m["key"])
        elif m["kind"] == "mouse_button":
            getattr(self._p, "mouseDown" if down else "mouseUp")(button=m["button"])

    def release_all(self) -> None:
        for i in sorted(self._held, reverse=True):
            self._press(self.mappings[i], False)
        self._held = set()

    def close(self) -> None:
        self.release_all()


def build_emitter(config: dict[str, Any], vg_module):
    actions = config["actions"]
    mappings, mode = actions["mappings"], actions.get("input_mode", "gamepad")
    validate_mappings(mappings, int(actions["num_actions"]))
    if mode == "gamepad":
        if vg_module is None:
            raise RuntimeError(
                "input_mode='gamepad' needs vgamepad and the ViGEmBus driver (Windows only). "
                "Use input_mode='keyboard_mouse', or run with dummy=True for offline training."
            )
        return GamepadEmitter(mappings, vg_module)
    if mode == "keyboard_mouse":
        return KeyboardMouseEmitter(mappings)
    raise ValueError(f"unknown actions.input_mode {mode!r}")
