"""
Generic configuration for any game
Adjust these variables for the game you want to train.
"""

# ==============================================
# GAME CONFIGURATION (EDIT HERE)
# ==============================================

GAME_CONFIG = {
    # Game or emulator process name
    # Examples: "rpcs3" (PS3), "pcsx2-qt" (PS2), "dolphin" (Wii/GameCube), 
    #           "re9" (Resident Evil Requiem), "hajime_no_ippo" (example)
    "process_name": "RobloxPlayerBeta",
    
    # Path to the game executable (optional - leave None if it is already open)
    # Example: R"C:\Games\RPCS3\rpcs3.exe"
    # Or for Steam: R"C:\Program Files (x86)\Steam\steam.exe -applaunch APPID"
    "exe_path": None,
    
    # Path to the game ROM
    "rom_path": None,
    
    # Screen capture settings
    "capture": {
        # Set this to the resolution the game will be played at
        # The model uses 128x128 internally; this is the raw captured resolution
        "width": 854,       # Capture width (e.g. 854 for 480p, 1280 for 720p)
        "height": 480,      # Capture height
        "internal_width": 128,    # Resolution the model processes
        "internal_height": 128,   # Resolution the model processes
        "target_fps": 60,       # Target capture FPS
        "buffer_len": 1,         # DXCam buffer (1 = faster, 3 = smoother)
    },
    
    # Window offset (adjust if needed to avoid emulator borders/menus)
    "window_offset": {
        "left": 20,     # pixels to the left of the window rect
        "top": 100,     # pixels from the top (skip the title bar, etc.)
        "right": 0,     # pixels to the right
        "bottom": 0,    # pixels from the bottom
    },
    
    # ===========================================
    # ACTION MAPPING (ADJUST PER GAME)
    # ===========================================
    # Adjust the actions to match your game.
    # The list length defines the number of actions.
    # Each action is binary (0 or 1).
    "actions": {
        # Number of actions (default: 9). Change here if needed.
        "num_actions": 9,
        
        # Whether the agent uses "gamepad" (Xbox controller) or "keyboard_mouse" (keyboard and mouse)
        "input_mode": "keyboard_mouse", 
        
        # Button mapping. To add more, add a new entry and raise num_actions.
        "mappings": [
            # WASD keys
            {"name": "W", "type": "key", "key": "w"},
            {"name": "A", "type": "key", "key": "a"},
            {"name": "S", "type": "key", "key": "s"},
            {"name": "D", "type": "key", "key": "d"},
            
            # Extra keys
            {"name": "SPACE", "type": "key", "key": "space"},
            {"name": "F", "type": "key", "key": "f"},
            {"name": "R", "type": "key", "key": "r"},
            
            # Mouse buttons
            {"name": "CLICK_L", "type": "mouse_button", "button": "left"},
            {"name": "CLICK_R", "type": "mouse_button", "button": "right"},
        ]
    }
}


# ==============================================
# TRAINING CONFIGURATION
# ==============================================

TRAINING_CONFIG = {
    # max trajectories per recording session (low cap to avoid MemoryError when saving)
    "max_trajectories": 3,
    
    # training batch size
    "batch_size": 384,
    
    # number of training epochs
    "epochs": 100,
    
    # learning rate
    "learning_rate": 1e-4,
    
    # temporal buffer size (LSTM window)
    "window_size": 10,
    
    # number of DAgger steps
    "dagger_iterations": 3,
    
    # directories
    "demo_path": "./demos/",
    "model_path": "./models/",
    "train_path": "./models/imitation/",
}


# ==============================================
# INPUT CONFIGURATION
# ==============================================

INPUT_CONFIG = {
    # Deadzone for analog sticks (0-1)
    "deadzone": 0.3,  # 30% of full stick travel
    
    # camera sensitivity (right stick multiplier)
    "camera_sensitivity": 1.0,
    
    # Delay between gamepad reads (seconds)
    "input_delay": 0.01,
    
    # AGGRESSIVENESS FACTOR (probability multiplier for mouse buttons)
    # 1.0 = Normal (agent only clicks when confident)
    # 2.0 = Aggressive (agent clicks at half the confidence needed)
    # 3.0 = Very aggressive (agent spams attacks at the slightest intent)
    "aggressiveness": 2.0,
}
