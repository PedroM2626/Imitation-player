"""
Profile: a generic PC title driven by keyboard and mouse (Roblox is the
example this configuration was written against). It has never been trained on:
the recorded corpus in ``runs/hajime_ippo/demos`` belongs to the Hajime
profile. See docs/DATA.md section 6.1.

``input_mode`` ``keyboard_mouse`` makes the environment emit synthesised key and
mouse events instead of driving a virtual gamepad.
"""

GAME_CONFIG = {
    "profile": "roblox",
    "process_name": "RobloxPlayerBeta",
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
        "input_mode": "keyboard_mouse",
        "num_actions": 9,
        "width_policy": "strict",
        "mappings": [
            {"name": "FWD", "kind": "key", "key": "w"},
            {"name": "BACK", "kind": "key", "key": "s"},
            {"name": "LEFT", "kind": "key", "key": "a"},
            {"name": "RIGHT", "kind": "key", "key": "d"},
            {"name": "JUMP", "kind": "key", "key": "space"},
            {"name": "ABILITY", "kind": "key", "key": "f"},
            {"name": "RELOAD", "kind": "key", "key": "r"},
            {"name": "ATTACK", "kind": "mouse_button", "button": "left"},
            {"name": "AIM", "kind": "mouse_button", "button": "right"},
        ],
    },
    "input": {"deadzone": 8000},
    "deploy": {"fps": 60, "aggressiveness": 1.0, "attack_buttons": ("ATTACK",)},
    "recording": {"max_trajectories": 10},
}

TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 384,
    "learning_rate": 1e-4,
    "window_size": 10,
    "dagger_iterations": 3,
}
