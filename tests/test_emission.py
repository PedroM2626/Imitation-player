"""Emission is driven by the mapping table, and conflicts cancel."""

import types

import numpy as np
import pytest

from agent.utils.emission import GamepadEmitter, KeyboardMouseEmitter, validate_mappings


class FakeGamepad:
    def __init__(self):
        self.pressed, self.released = set(), set()
        self.left = (0.0, 0.0)
        self.right = (0.0, 0.0)
        self.lt = 0
        self.rt = 0
        self.updates = 0

    def press_button(self, b):
        self.pressed.add(b)

    def release_button(self, b):
        self.released.add(b)

    def left_joystick_float(self, x_value_float, y_value_float):
        self.left = (x_value_float, y_value_float)

    def right_joystick_float(self, x_value_float, y_value_float):
        self.right = (x_value_float, y_value_float)

    def left_trigger(self, value):
        self.lt = value

    def right_trigger(self, value):
        self.rt = value

    def update(self):
        self.updates += 1


def make_vg():
    """A vgamepad-shaped stub exposing XUSB_BUTTON constants and a pad factory."""
    buttons = types.SimpleNamespace(
        XUSB_GAMEPAD_A=1, XUSB_GAMEPAD_B=2, XUSB_GAMEPAD_X=4, XUSB_GAMEPAD_L3=8)

    def factory():
        return FakeGamepad()

    return types.SimpleNamespace(XUSB_BUTTON=buttons, VX360Gamepad=factory)


MAPPINGS = [
    {"name": "UP", "kind": "axis", "stick": "left", "axis": "y", "value": -1.0},
    {"name": "DOWN", "kind": "axis", "stick": "left", "axis": "y", "value": 1.0},
    {"name": "CROSS", "kind": "button", "button": "B"},
    {"name": "L2", "kind": "trigger", "trigger": "left", "value": 255},
    {"name": "CAM_R", "kind": "axis", "stick": "right", "axis": "x", "value": 0.5},
    {"name": "CAM_R_FAST", "kind": "axis", "stick": "right", "axis": "x", "value": 1.0},
]


def emitter():
    e = GamepadEmitter(MAPPINGS, make_vg())
    return e


def test_validate_rejects_width_mismatch():
    with pytest.raises(ValueError, match="must be equal"):
        validate_mappings(MAPPINGS, 18)


def test_validate_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        validate_mappings([{"name": "X", "kind": "wobble"}], 1)


def test_opposing_axis_bits_cancel():
    """The old code let the last-tested index win; now they sum and clamp."""
    e = emitter()
    e.apply(np.array([1, 1, 0, 0, 0, 0], dtype=np.float32))  # UP + DOWN
    assert e.gamepad.left[1] == pytest.approx(0.0)


def test_camera_axis_accumulates_and_clamps():
    e = emitter()
    e.apply(np.array([0, 0, 0, 0, 1, 1], dtype=np.float32))  # 0.5 + 1.0 -> clamp 1.0
    assert e.gamepad.right[0] == pytest.approx(1.0)


def test_button_press_and_release_are_symmetric():
    e = emitter()
    e.apply(np.array([0, 0, 1, 0, 0, 0], dtype=np.float32))
    assert e.gamepad.pressed == {2}
    e.apply(np.zeros(6, dtype=np.float32))
    assert e.gamepad.released == {2}


def test_trigger_value_follows_the_bit():
    e = emitter()
    e.apply(np.array([0, 0, 0, 1, 0, 0], dtype=np.float32))
    assert e.gamepad.lt == 255
    e.apply(np.zeros(6, dtype=np.float32))
    assert e.gamepad.lt == 0


def test_every_step_updates_the_device():
    e = emitter()
    e.apply(np.zeros(6, dtype=np.float32))
    e.apply(np.zeros(6, dtype=np.float32))
    assert e.gamepad.updates == 2


class FakePyDirect:
    FAILSAFE = True

    def __init__(self):
        self.events = []

    def keyDown(self, k):
        self.events.append(("down", k))

    def keyUp(self, k):
        self.events.append(("up", k))

    def mouseDown(self, button):
        self.events.append(("down", button))

    def mouseUp(self, button):
        self.events.append(("up", button))


def test_keyboard_mouse_release_all(monkeypatch):
    stub = FakePyDirect()
    monkeypatch.setitem(__import__("sys").modules, "pydirectinput", stub)
    e = KeyboardMouseEmitter([{"name": "FWD", "kind": "key", "key": "w"},
                              {"name": "ATK", "kind": "mouse_button", "button": "left"}])
    e.apply(np.array([1, 1], dtype=np.float32))
    e.release_all()
    assert stub.events.count(("up", "w")) == 1
    assert stub.events.count(("up", "left")) == 1
