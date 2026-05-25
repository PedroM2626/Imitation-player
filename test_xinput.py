import ctypes
import time

xinput = ctypes.windll.xinput1_4

class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ('wButtons', ctypes.c_ushort),
        ('bLeftTrigger', ctypes.c_ubyte),
        ('bRightTrigger', ctypes.c_ubyte),
        ('sThumbLX', ctypes.c_short),
        ('sThumbLY', ctypes.c_short),
        ('sThumbRX', ctypes.c_short),
        ('sThumbRY', ctypes.c_short)
    ]

class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ('dwPacketNumber', ctypes.c_ulong),
        ('Gamepad', XINPUT_GAMEPAD)
    ]

state = XINPUT_STATE()
for i in range(4):
    res = xinput.XInputGetState(i, ctypes.byref(state))
    print(f'Controller {i}: res={res}, buttons={state.Gamepad.wButtons}, LX={state.Gamepad.sThumbLX}, LY={state.Gamepad.sThumbLY}')
