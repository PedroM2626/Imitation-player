"""
Human-input capture for recording and for DAgger corrections.

This is the single place that turns a real device state into the profile's
action vector, so a demonstration, a correction and a manual play session all
mean the same thing.

Two problems in the previous implementation are fixed here:

* the recorder only ever wrote indices 0-6, so triggers, stick press and the
  camera axes were *unrecordable* no matter how the operator played. Every bit
  of the gamepad table now has a physical source;
* the DAgger script and the recorder carried separate copies of this logic and
  had drifted (one polled XInput, the other only the keyboard).
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

import numpy as np

XINPUT_BUTTON_MASKS = {
    "DPAD_UP": 0x0001,
    "DPAD_DOWN": 0x0002,
    "DPAD_LEFT": 0x0004,
    "DPAD_RIGHT": 0x0008,
    "START": 0x0010,
    "BACK": 0x0020,
    "L3": 0x0040,
    "R3": 0x0080,
    "LB": 0x0100,
    "RB": 0x0200,
    "A": 0x1000,
    "B": 0x2000,
    "X": 0x4000,
    "Y": 0x8000,
    "GUIDE": 0x40000,
}

_STICK_FIELDS = {
    ("left", "x"): "sThumbLX",
    ("left", "y"): "sThumbLY",
    ("right", "x"): "sThumbRX",
    ("right", "y"): "sThumbRY",
}

# Keyboard stand-ins, used when no physical pad is connected. A profile may
# override any of these with input.keyboard_keys = {"<mapping name>": "<key>"}.
DEFAULT_KEYBOARD_KEYS = {
    "UP": "up",
    "DOWN": "down",
    "LEFT": "left",
    "RIGHT": "right",
    "CROSS": "i",
    "CIRCLE": "o",
    "SQUARE": "p",
    "TRIANGLE": "u",
    "L2": "j",
    "R2": "k",
    "L3": "l",
    "R3": "semicolon",
    "CAM_RIGHT": "right",
    "CAM_LEFT": "left",
}


class _XInputGamepad(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class _XInputState(ctypes.Structure):
    _fields_ = [("dwPacketNumber", ctypes.c_ulong), ("Gamepad", _XInputGamepad)]


class XInputState:
    """ctypes binding for XInput1_4/1_3, absent on non-Windows platforms."""

    def __init__(self, user_index: int = 0) -> None:
        self.user_index = user_index
        self._lib = None
        try:
            self._lib = ctypes.windll.xinput1_4  # type: ignore[attr-defined]
        except (AttributeError, OSError):
            try:
                self._lib = ctypes.windll.xinput1_3  # type: ignore[attr-defined]
            except (AttributeError, OSError):
                self._lib = None

    @property
    def available(self) -> bool:
        return self._lib is not None

    def read(self) -> _XInputGamepad | None:
        if self._lib is None:
            return None
        state = _XInputState()
        if self._lib.XInputGetState(self.user_index, ctypes.byref(state)) == 0:
            return state.Gamepad
        return None


def _gamepad_predicate(mapping: dict[str, Any], input_cfg: dict[str, Any]) -> Callable[[Any], bool]:
    kind = mapping["kind"]
    if kind == "button":
        mask = XINPUT_BUTTON_MASKS[mapping["button"]]
        return lambda g: bool(g.wButtons & mask)

    if kind == "trigger":
        field = "bLeftTrigger" if mapping["trigger"] == "left" else "bRightTrigger"
        threshold = int(input_cfg.get("trigger_threshold", 30))
        return lambda g: getattr(g, field) >= threshold

    if kind == "axis":
        field = _STICK_FIELDS[(mapping["stick"], mapping["axis"])]
        value = float(mapping["value"])
        sign = 1 if value > 0 else -1
        magnitude = abs(value)
        if mapping["stick"] == "left":
            threshold = int(input_cfg.get("deadzone", 8000))
            dpad = {
                ("left", "y", -1): "DPAD_UP",
                ("left", "y", 1): "DPAD_DOWN",
                ("left", "x", -1): "DPAD_LEFT",
                ("left", "x", 1): "DPAD_RIGHT",
            }
            pad_mask = XINPUT_BUTTON_MASKS.get(
                dpad.get((mapping["stick"], mapping["axis"], sign), ""), 0
            )
            return lambda g: (
                (sign * getattr(g, field) > threshold) or (pad_mask and bool(g.wButtons & pad_mask))
            )
        # Right stick: the +/-0.5 flags fire in the half band, the +/-1.0 flags above it.
        half = int(input_cfg.get("camera_half_deadzone", 8000))
        full = int(input_cfg.get("camera_full_deadzone", 20000))
        ceiling = full if magnitude >= 1.0 else half
        return lambda g: sign * getattr(g, field) > ceiling

    raise ValueError(
        f"mapping {mapping.get('name')!r} has kind {kind!r}, which is not a gamepad control"
    )


class HumanInput:
    """Reads one action vector per call, from the device the profile asks for."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        actions = config["actions"]
        self.mappings: list[dict[str, Any]] = actions["mappings"]
        self.num_actions = int(actions["num_actions"])
        self.mode = actions.get("input_mode", "gamepad")
        self.input_cfg = config.get("input", {})
        self.pad = XInputState()
        self._predicates = [
            _gamepad_predicate(m, self.input_cfg)
            for m in self.mappings
            if m["kind"] in ("button", "trigger", "axis")
        ]
        self._keyboard = self._load_keyboard()

    def _load_keyboard(self):
        try:
            import keyboard

            return keyboard
        except Exception:  # noqa: BLE001 - needs root on Linux, absent deps elsewhere
            return None

    def _key_bindings(self) -> dict[int, str]:
        overrides = dict(DEFAULT_KEYBOARD_KEYS)
        overrides.update(self.input_cfg.get("keyboard_keys", {}))
        bindings = {}
        for i, m in enumerate(self.mappings):
            key = overrides.get(m.get("name", ""))
            if key:
                bindings[i] = key
        return bindings

    def read(self) -> np.ndarray:
        action = np.zeros(self.num_actions, dtype=np.float32)

        if self.mode == "keyboard_mouse":
            return self._read_keyboard_mouse(action)

        gamepad_bits = [
            i for i, m in enumerate(self.mappings) if m["kind"] in ("button", "trigger", "axis")
        ]
        state = self.pad.read()
        if state is not None:
            for bit, predicate in zip(gamepad_bits, self._predicates, strict=True):
                if predicate(state):
                    action[bit] = 1.0
            return action

        # No pad connected: fall back to the keyboard so recording still works.
        if self._keyboard is None:
            return action
        for bit, key in self._key_bindings().items():
            if self._keyboard.is_pressed(key):
                action[bit] = 1.0
        return action

    def _read_keyboard_mouse(self, action: np.ndarray) -> np.ndarray:
        if self._keyboard is None:
            return action
        try:
            import mouse
        except Exception:  # noqa: BLE001
            mouse = None
        for i, m in enumerate(self.mappings):
            pressed_key = m["kind"] == "key" and self._keyboard.is_pressed(m["key"])
            pressed_button = (
                m["kind"] == "mouse_button" and mouse is not None and mouse.is_pressed(m["button"])
            )
            if pressed_key or pressed_button:
                action[i] = 1.0
        return action
