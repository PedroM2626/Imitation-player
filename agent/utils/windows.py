"""
Platform-optional imports.

The capture and actuation stack is Windows-only, but *training* is not: it
consumes recorded demonstrations and never touches the OS. Keeping the
platform modules behind these flags is what makes ``dummy=True`` environments
importable on Linux and on Windows machines without the ViGEmBus driver --
which is what the Dockerfile and the test suite rely on.
"""

from __future__ import annotations

import sys

IS_WINDOWS = sys.platform == "win32"

win32gui = win32process = None
dxcam = None
vg = None
mss = None

HAS_WIN32 = HAS_DXCAM = HAS_VGAMEPAD = HAS_MSS = False

if IS_WINDOWS:
    try:
        import win32gui  # type: ignore  # noqa: F401 - re-exported
        import win32process  # type: ignore  # noqa: F401 - re-exported

        HAS_WIN32 = True
    except ImportError:
        pass
    try:
        import dxcam  # type: ignore  # noqa: F401 - re-exported

        HAS_DXCAM = True
    except (ImportError, OSError):
        pass
    try:
        import vgamepad as vg  # type: ignore  # noqa: F401 - re-exported

        HAS_VGAMEPAD = True
    except (ImportError, OSError, Exception):  # noqa: BLE001 - VBus raises bare Exception
        HAS_VGAMEPAD = False

try:
    import mss  # type: ignore  # noqa: F401 - re-exported

    HAS_MSS = True
except (ImportError, Exception):  # noqa: BLE001
    HAS_MSS = False


def require_desktop_access(feature: str) -> None:
    """Raise a pointed error when a live-capture feature is unavailable."""
    if not IS_WINDOWS:
        raise RuntimeError(
            f"{feature} needs Windows; run training with a dummy environment instead "
            f"(see agent.utils.game_env, config['dummy']=True)."
        )
    raise RuntimeError(
        f"{feature} is unavailable: the required package or driver is not installed."
    )
