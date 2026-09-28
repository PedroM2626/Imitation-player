"""
Copy this file to ``local.py`` (same directory, git-ignored) and fill in the
paths that exist only on your machine. Values here override the profile, and
only the keys listed in ``agent.config._LOCAL_KEYS`` are honoured.

    from agent.config.local import LOCAL_OVERRIDES   # not imported by the loader
    # The loader imports agent.config.local automatically when it exists.
"""

LOCAL_OVERRIDES = {
    "hajime_ippo": {
        "exe_path": r"C:\emuladores\rpcs3\rpcs3.exe",
        "rom_path": r"C:\roms\Hajime no Ippo - The Fighting! (Japan).iso",
    },
    "roblox": {
        # Leave exe_path None when you start the game yourself; the environment
        # then only waits for the window to appear.
        "exe_path": None,
        "rom_path": None,
    },
}
