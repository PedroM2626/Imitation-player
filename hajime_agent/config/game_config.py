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
    "process_name": "rpcs3",
    
    # Path to the game executable (optional - leave None if it is already open)
    # Example: R"C:\Games\RPCS3\rpcs3.exe"
    # Or for Steam: R"C:\Program Files (x86)\Steam\steam.exe -applaunch APPID"
    "exe_path": R"D:\emuladores\rpcs3-v0.0.39-18737-818b11fd_win64_msvc\rpcs3.exe",
    
    # Path to the game ROM
    "rom_path": R"D:\roms\Hajime no Ippo - The Fighting! (Japan).iso",
    
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
        # Number of actions (default: 18). Change here if needed.
        "num_actions": 18,
        
        # Button mapping for the virtual gamepad
        # Available types: "button", "axis" (analog stick), "trigger"
        "mappings": [
            # Example: D-Pad / left stick
            {"name": "UP", "type": "axis", "vg_code": "left_joystick", "axis": "y", "value": -1.0},
            {"name": "DOWN", "type": "axis", "vg_code": "left_joystick", "axis": "y", "value": 1.0},
            {"name": "LEFT", "type": "axis", "vg_code": "left_joystick", "axis": "x", "value": -1.0},
            {"name": "RIGHT", "type": "axis", "vg_code": "left_joystick", "axis": "x", "value": 1.0},
            
            # Face buttons
            {"name": "CROSS", "type": "button", "vg_code": "DS4_BUTTON_CROSS"},      # X on PlayStation
            {"name": "CIRCLE", "type": "button", "vg_code": "DS4_BUTTON_CIRCLE"},    # O on PlayStation
            {"name": "SQUARE", "type": "button", "vg_code": "DS4_BUTTON_SQUARE"},     # Square on PlayStation
            
            # Triggers (in fighting games these can act as dodge/parry)
            {"name": "L2", "type": "trigger", "vg_code": "left_trigger", "value": 255},   # Left trigger
            {"name": "R2", "type": "trigger", "vg_code": "right_trigger", "value": 255},  # Right trigger
            
            # L3 (left stick press) - can be dash/run
            {"name": "L3", "type": "button", "vg_code": "DS4_BUTTON_THUMB_LEFT"},
            
            # Right stick X (camera / dodge) - discretized into 4 directions
            {"name": "CAM_RIGHT", "type": "axis", "vg_code": "right_joystick", "axis": "x", "value": 0.5},
            {"name": "CAM_RIGHT_FAST", "type": "axis", "vg_code": "right_joystick", "axis": "x", "value": 1.0},
            {"name": "CAM_LEFT", "type": "axis", "vg_code": "right_joystick", "axis": "x", "value": -0.5},
            {"name": "CAM_LEFT_FAST", "type": "axis", "vg_code": "right_joystick", "axis": "x", "value": -1.0},
            
            # Right stick Y
            {"name": "CAM_UP", "type": "axis", "vg_code": "right_joystick", "axis": "y", "value": -0.5},
            {"name": "CAM_UP_FAST", "type": "axis", "vg_code": "right_joystick", "axis": "y", "value": -1.0},
            {"name": "CAM_DOWN", "type": "axis", "vg_code": "right_joystick", "axis": "y", "value": 0.5},
            {"name": "CAM_DOWN_FAST", "type": "axis", "vg_code": "right_joystick", "axis": "y", "value": 1.0},
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
}
