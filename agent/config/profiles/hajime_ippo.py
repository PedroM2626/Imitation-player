"""
Profile: Hajime no Ippo - The Fighting! (PS3 / RPCS3), played with a gamepad.

Machine-specific paths (``exe_path``, ``rom_path``) are intentionally ``None``
here: put them in ``agent/config/local.py`` so this file stays portable.

Action semantics
----------------
The vector is ``MultiBinary(18)``. Indices 0-6 are the directions and face
buttons actually used by the recorded corpus; indices 7-17 (triggers, stick
press, camera) are *reachable* through the physical-pad reader in
``agent.utils.input_map``, but were never actuated in the demonstrations that
exist today -- see docs/DATA.md.

Every entry in ``actions.mappings`` drives both capture and emission, so the
table is the single source of truth. ``kind`` is one of:

    button   -> a digital XInput button            (``button``)
    axis     -> a stick direction, value in [-1,1] (``stick``, ``axis``, ``value``)
    trigger  -> an analog trigger held fully down  (``trigger``, ``value``)
"""

GAME_CONFIG = {
    "profile": "hajime_ippo",
    "process_name": "rpcs3",
    "exe_path": None,
    "rom_path": None,
    "hide_window": False,
    "capture": {
        "internal_width": 128,
        "internal_height": 128,
        "target_fps": 60,
        "buffer_len": 1,
    },
    "window_offset": {"left": 20, "top": 100, "right": 0, "bottom": 0},
    "actions": {
        "input_mode": "gamepad",
        "num_actions": 18,
        # What to do when a recorded demo has a different width: "strict"
        # refuses to train, "coerce" truncates/pads and prints a warning.
        "width_policy": "strict",
        "mappings": [
            {"name": "UP", "kind": "axis", "stick": "left", "axis": "y", "value": -1.0},
            {"name": "DOWN", "kind": "axis", "stick": "left", "axis": "y", "value": 1.0},
            {"name": "LEFT", "kind": "axis", "stick": "left", "axis": "x", "value": -1.0},
            {"name": "RIGHT", "kind": "axis", "stick": "left", "axis": "x", "value": 1.0},
            {"name": "CROSS", "kind": "button", "button": "B"},      # PlayStation Cross = Xbox B
            {"name": "CIRCLE", "kind": "button", "button": "A"},     # PlayStation Circle = Xbox A
            {"name": "SQUARE", "kind": "button", "button": "X"},
            {"name": "L2", "kind": "trigger", "trigger": "left", "value": 255},
            {"name": "R2", "kind": "trigger", "trigger": "right", "value": 255},
            {"name": "L3", "kind": "button", "button": "L3"},
            {"name": "CAM_RIGHT", "kind": "axis", "stick": "right", "axis": "x", "value": 0.5},
            {"name": "CAM_RIGHT_FAST", "kind": "axis", "stick": "right", "axis": "x", "value": 1.0},
            {"name": "CAM_LEFT", "kind": "axis", "stick": "right", "axis": "x", "value": -0.5},
            {"name": "CAM_LEFT_FAST", "kind": "axis", "stick": "right", "axis": "x", "value": -1.0},
            {"name": "CAM_UP", "kind": "axis", "stick": "right", "axis": "y", "value": -0.5},
            {"name": "CAM_UP_FAST", "kind": "axis", "stick": "right", "axis": "y", "value": -1.0},
            {"name": "CAM_DOWN", "kind": "axis", "stick": "right", "axis": "y", "value": 0.5},
            {"name": "CAM_DOWN_FAST", "kind": "axis", "stick": "right", "axis": "y", "value": 1.0},
        ],
    },
    "input": {
        # Analog stick thresholds, in raw XInput units (-32768..32767).
        "deadzone": 8000,
        # Camera-axis stick thresholds: |value| > half -> the *_FAST variant.
        "camera_half_deadzone": 8000,
        "camera_full_deadzone": 20000,
        # Trigger press threshold, raw 0..255.
        "trigger_threshold": 30,
    },
    "deploy": {
        # Target inference steps per second. 30 is deliberately conservative:
        # the emulator renders at its own rate and the policy is only reading.
        "fps": 30,
        # Probability multiplier applied to attack-like buttons at inference.
        # 1.0 disables the sharpening path entirely.
        "aggressiveness": 1.0,
        "attack_buttons": ("CROSS", "CIRCLE", "SQUARE"),
    },
    "recording": {"max_trajectories": 10},
}

TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 384,
    "learning_rate": 1e-4,
    "window_size": 10,
    "dagger_iterations": 3,
}
