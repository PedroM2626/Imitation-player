# Setup

> Getting a clean clone of this repository to a running training job, and an explicit account of
> what does not work. Architecture rationale lives in [ARCHITECTURE.md](ARCHITECTURE.md); the
> operator-facing pipeline lives in [TRAINING_GUIDE.md](TRAINING_GUIDE.md). Every claim below was
> checked against the source in this working copy; where the top-level README disagrees, the
> disagreement is stated.

## 1. Platform reality: the environment layer is Windows-only

`*/utils/game_env.py` is imported by every script in `notebooks/`, and its module-level imports are
identical in both packages (`generic_agent/utils/game_env.py:6-21`, `hajime_agent/utils/game_env.py:6-21`):

| Line | Import | Why it pins the platform |
|---|---|---|
| 8 | `import win32gui` | `pywin32`; window enumeration, `IsWindowVisible`, `GetWindowRect`, `MoveWindow` |
| 10 | `import win32process` | `pywin32`; `GetWindowThreadProcessId` maps a window to a PID |
| 12 | `import psutil` | portable, but used only to resolve the PID to a process name |
| 13 | `import vgamepad as vg` | ViGEmBus virtual Xbox 360 pad (`vg.VX360Gamepad()`) |
| 19 | `import dxcam` | Windows Desktop Duplication API (GPU-side screen grab) |

`mss` (line 151, and line 138 in the `hajime_agent` copy) and `pydirectinput` (line 132,
`generic_agent` only) are imported *inside* `__init__`, so they are optional at import time but still
Windows-bound at runtime.
`keyboard` is a module-level import in every `run_ai*.py`, `run_dagger.py` and
`record_trajectories.py`; `mouse` is imported lazily inside the `keyboard_mouse` branches.
`test_xinput.py:4` does `ctypes.windll.xinput1_4`, which exists only on Windows.

**Consequence for offline work.** `GenericGameEnv` accepts `config["dummy"] = True` and returns from
`__init__` at line 103 before any window, pad or camera is created — but that early return happens
*after* the module-level imports have already executed. There is therefore **no way to import
`game_env.py` on Linux**, even for dummy/offline training, and no way to import it on Windows without
ViGEmBus: `vgamepad/__init__.py` imports `vgamepad.win.virtual_gamepad`, which instantiates the module
global `VBUS = VBus()` at import time, and `VBus.__init__` calls `check_err(vigem_connect(...))`, which
raises `Exception` on any non-`VIGEM_ERROR_NONE` return. A CPU-only, headless training path would
require moving lines 8, 10, 13 and 19 into the non-dummy branch.

## 2. Hardware

- **GPU.** Any run of practical length needs an NVIDIA GPU with CUDA 12.x. The frozen environment
  pins `torch==2.5.1+cu121` (`requirements.txt:125`).
- **What degrades without CUDA.** All training scripts default to `--device cuda`
  (`train_agent.py:393`), and that default is *not* repaired automatically: `train_agent.py:401` reads
  `device = args.device or check_cuda()`, so because the argparse default is the non-empty string
  `"cuda"`, `check_cuda()` is never reached and PyTorch raises. Only `compare_models.py:421-423`
  performs a real fallback. Pass `--device cpu` explicitly on CPU machines and expect the 10-epoch
  benchmark (18 s – 55 min per encoder on GPU, README §10.1) to become impractical.
- **VRAM.** No memory measurement exists anywhere in the repository (no `torch.cuda.max_memory_*`,
  no `nvidia-smi` invocation — verified by grep). The following is arithmetic on the constructor
  defaults, not a measurement. With `--batch 384` over `(4, 128, 128)` float input, one input
  activation is `384 × 4 × 128 × 128 × 4 B ≈ 96 MiB`. The memory hotspot is the Impala-CNN flatten
  head: `ImpalaCNNExtractor` computes `flatten_dim = (128 // 2**3) × (128 // 2**3) × 128 = 16 × 16 ×
  128 = 32 768` (`utils/new_architectures.py:73-75`) and builds `Linear(32768 → 512)` (line 79), i.e.
  16 777 216 weights ≈ 64 MiB of parameters, plus ≈ 64 MiB of gradients and ≈ 128 MiB of Adam
  moments for that single layer, plus a 48 MiB flatten activation per minibatch. The GAP-based
  `ImpoolaCNNExtractor` replaces that input with `AdaptiveAvgPool2d((1,1))`, so its head is
  `Linear(128 → 512)`. Recorded artefact sizes track this: 67.95 MB (Impala-CNN) vs 4.20 MB
  (Impoola-CNN) — README §10.1.
- **Observed parameter counts**, printed by `compare_models.py:210` as the MLflow metric `num_params`:
  NatureCNN 4 196 810 · CNN+LSTM+Attention 6 116 779 · ViT 2 448 010 · Impoola-CNN 1 009 258 ·
  Impala-CNN 17 720 938 · ResNet-18 11 516 938 (README §10.1).

## 3. Python

Python **3.11**. Evidence: `requirements.txt` contains only `cp311`-compatible pins, and the compiled
modules still present under `generic_agent/utils/__pycache__/` are `*.cpython-311.pyc`.

**The `venv/` directory present in this working copy is broken and should be deleted.**
`venv/pyvenv.cfg` records

```
home = C:\Users\pedro\AppData\Local\Programs\Python\Python311
executable = C:\Users\pedro\AppData\Local\Programs\Python\Python311\python.exe
version = 3.11.9
```

That interpreter does not exist on the machine this repository now sits on, so `venv\Scripts\python.exe`
is a stub pointing at a missing base. `venv/` is listed in `.gitignore`, so this is local litter rather
than a repository defect — but it will shadow a correct environment if left in place.

```powershell
Remove-Item -Recurse -Force venv
python -m venv venv
venv\Scripts\Activate.ps1
```

## 4. Install path A — the frozen environment

```powershell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

`requirements.txt` is a **verbatim `pip freeze` of one workstation: 142 pinned packages**, not a
curated dependency list. It therefore carries MLflow's whole server stack (Flask, alembic, graphene,
waitress, opentelemetry), `tensorboard`, `optuna`, `datasets`, `pygame`, `comtypes`, and a `mouse` /
`keyboard` / `inputs` trio of input libraries. It also pins three packages with **local version
labels** — `torch==2.5.1+cu121`, `torchaudio==2.5.1+cu121`, `torchvision==0.20.1+cu121` — which do not
exist on PyPI and resolve only from PyTorch's CUDA index. If `--extra-index-url` does not satisfy you,
install the three wheels first:

```powershell
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt --no-deps
```

The README's minimal-install snippet (§13.2) is stale on two counts: it says `gymnasium==0.29.2`
(the file pins **0.29.1**) and omits `pywin32`'s and `comtypes`' pins.

## 5. Install path B — curated minimum

Derived by grepping every `import` statement in `generic_agent/`, `hajime_agent/` and
`test_xinput.py`; versions are the ones in `requirements.txt`.

```powershell
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium==0.29.1 stable-baselines3==2.2.1 imitation==1.0.1 numpy==2.4.4 opencv-python==4.13.0.92 mlflow==3.12.0 psutil==7.2.2 pygame==2.6.1
pip install pywin32==311 vgamepad==0.1.0 dxcam==0.3.0 comtypes==1.4.16 keyboard==0.13.5 mss==10.2.0 PyDirectInput==1.0.4 mouse==0.7.1
```

| Package | Imported by |
|---|---|
| `torch`, `torchvision` | every `utils/*.py`; `torchvision.models.resnet18` in `new_architectures.py:12` |
| `gymnasium`, `stable-baselines3` | environment and policy classes |
| `imitation` | `imitation.algorithms.BC`, `imitation.data.types.Trajectory`, `imitation.GAIL` |
| `numpy`, `opencv-python` | array handling and `cv2.resize` / colour conversion |
| `mlflow` | explicit import in the training scripts; **not** a declared dependency of `imitation==1.0.1`, so it must be installed by hand |
| `psutil`, `pywin32`, `vgamepad`, `dxcam` | the Windows capture/actuation layer |
| `keyboard`, `pygame` | hotkeys and the preview window in `record_trajectories.py` / `run_ai*.py` |
| `mss`, `PyDirectInput`, `mouse` | lazily: `mss` on DXCam failure, the other two in `keyboard_mouse` mode |
| `comtypes` | dependency of `dxcam`; imported by no repository module |

`inputs==0.5` appears in `requirements.txt:55` but is imported by **nothing** in the tree (verified by
grep) — it is dead weight in the frozen file, and it is one of the five lines the `Dockerfile` strips.

## 6. ViGEmBus

`vgamepad` is a Python binding for the **ViGEmBus** kernel driver; the pip package does not install the
driver. Install it from the ViGEm repository (`ViGEmBus_Setup_x64.exe`) and reboot.

Two failure modes, in order of severity:

1. **Driver absent → `import vgamepad` raises** at module load (see §1), which surfaces as a traceback
   mentioning a `VIGEM_ERRORS` name (`VIGEM_ERROR_BUS_NOT_FOUND_ERROR` in practice) rather than an
   `ImportError`. This is loud.
2. **Driver present but the game ignores the pad** — some titles poll only DirectInput, or were started
   before the virtual device was attached. Restart the title after the pad exists.

The README's §13.3 claim that "the virtual pad is created but no game receives input, and the failure
is silent" understates case 1: with the driver missing, `import vgamepad` does not silently succeed.
For the record, `Get-Service ViGEmBus` reports the service **absent** on the machine this document was
written on, so no deploy or recording command in this repository can currently be executed there.

## 7. Game and emulator prerequisites

- **Start the title before constructing the environment.** `find_window_by_process_name`
  (`game_env.py:322-337`) enumerates visible top-level windows and matches the owning process name
  case-insensitively against `GAME_CONFIG["process_name"]`, returning the **first** hit in enumeration
  order. If no window is found and `exe_path` is set, `__init__` does `Popen([exe_path, rom_path])`
  (`game_env.py:111-120`), then `wait_start()` polls for one second at a time for **up to 120 seconds**
  (`game_env.py:339-361`). On timeout it only prints `WARNING: Window for '<name>' not found after 120s!`
  and returns — the object stays alive with `self.hwnd = None`, so every later capture yields the
  black/previous-frame fallback. **Check for the `Window founded HWND:` line before trusting a recording.**
- **Window offsets.** The grab rectangle is the window rect plus `window_offset.left/top` and minus
  `window_offset.right/bottom` (`_get_window_region`, `game_env.py:157-165`). The committed defaults
  (`left: 20, top: 100`) exist to skip an emulator's title bar and border. Adjust them per machine, or
  run the game borderless/fullscreen and zero them.
- **Absolute emulator paths must be edited locally.** `hajime_agent/config/game_config.py:19-22`
  hard-codes `"exe_path": R"D:\emuladores\rpcs3-v0.0.39-18737-818b11fd_win64_msvc\rpcs3.exe"` and
  `"rom_path": R"D:\roms\Hajime no Ippo - The Fighting! (Japan).iso"`. These are one user's paths and
  will not exist elsewhere. `generic_agent/config/game_config.py:19-22` sets both to `None`, i.e. the
  generic package requires the target window to be already open.
- **`keyboard_mouse` mode** (the `generic_agent` example config) additionally requires
  `actions.input_mode == "keyboard_mouse"` and a mapping table whose entries use the keys `type`,
  `key`/`button` — a different schema from the gamepad table (§TRAINING_GUIDE 2).

## 8. Verification, in order

Every script inserts `".."` and `"../utils"` into `sys.path` (`train_agent.py:48-49`,
`record_trajectories.py:20-21`, `run_ai.py:9-10`) and hard-codes `./demos/`, `./models/` and
`file:../mlruns` relative to the **current working directory**. All commands below must therefore be
run from inside `<pkg>/notebooks`, with the repository root as the root of the paths shown.

| # | Command (PowerShell) | Expected observable outcome |
|---|---|---|
| 1 | `python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"` | Python 3.11.x, `2.5.1+cu121`, `True`, `12.1`. Confirms the venv is active, the CUDA wheel resolved, and the driver matches the wheel |
| 2 | `cd generic_agent\notebooks`<br>`python -c "from config.game_config import GAME_CONFIG, TRAINING_CONFIG, INPUT_CONFIG; print(GAME_CONFIG['actions']['num_actions'], GAME_CONFIG['actions'].get('input_mode'))"` | `9 keyboard_mouse` for `generic_agent`; `18 None` for `hajime_agent` (that config declares no `input_mode`, and the environment's `.get(..., "gamepad")` default supplies it). Proves the `sys.path` layout resolves |
| 3 | `python -c "import os,sys; sys.path.insert(0,os.path.abspath('..')); sys.path.insert(0,os.path.abspath('../utils')); from game_env import GenericGameEnv; from config.game_config import GAME_CONFIG; cfg=dict(GAME_CONFIG); cfg['dummy']=True; e=GenericGameEnv(cfg); print(e.observation_space, e.action_space)"` | a `Box` of shape `(128, 128, 1)` dtype `uint8` and `MultiBinary(9)`. This is the real test that ViGEmBus is installed, because `import vgamepad` connects at import time. No game required |
| 4 | `python -c "import os,sys; sys.path.insert(0,os.path.abspath('..')); sys.path.insert(0,os.path.abspath('../utils')); from game_env import GenericGameEnv; from config.game_config import GAME_CONFIG; from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack; cfg=dict(GAME_CONFIG); cfg['dummy']=True; v=VecFrameStack(VecTransposeImage(DummyVecEnv([lambda: GenericGameEnv(cfg)])), n_stack=4); print(v.observation_space, v.action_space)"` | `Box(0, 255, (4, 128, 128), uint8)`. This is the exact wrapper stack the trainers build (`train_agent.py:277-285`, `compare_models.py:140-147`), i.e. the space every encoder is constructed against |
| 5 | `cd ..\..`<br>`python test_xinput.py` | four lines, `Controller i: res=..., buttons=..., LX=..., LY=...`. Observed on the reference machine with no pad attached: `res=1167` (`ERROR_DEVICE_NOT_CONNECTED`) on all four slots. Slot *N* returning `res=0` after a training run started means the virtual pad exists. **This probe does not exercise ViGEmBus by itself** — it only reads XInput slots, so `res=1167` everywhere is not evidence of a missing driver |
| 6 | `cd generic_agent\notebooks`<br>`python train_agent.py --epochs 1 --batch 32 --lr 1e-4 --device cpu` | `1. LOADING DATA`, a per-file load list, an action-distribution histogram, then one epoch of progress and `[OK] Model saved to: ./models/bc_policy.zip`. Requires at least one `demo*.pt` in `./demos/`; with an empty directory it raises `FileNotFoundError: No demo files found in demos/`, which is the correct outcome for a fresh clone |

Step 6 writes into `./models/` and `../mlruns`; delete `models/bc_policy.zip` afterwards if you intend
to benchmark, since `run_ai.py` prefers that file over any checkpoint (§TRAINING_GUIDE 8).

## 9. Docker: the `Dockerfile` does not work

`Dockerfile` cannot produce a runnable image. Findings, in build order:

| Line | Problem |
|---|---|
| 2 | Base is `nvidia/cuda:12.1.1-runtime-ubuntu22.04` — the **runtime** flavour, which ships the CUDA runtime libraries but no Python toolchain |
| 5-13 | `apt-get install -y python3.11` has no candidate in Ubuntu 22.04's default archives (jammy packages CPython 3.10; 3.11 needs the deadsnakes PPA, which the file never adds) → the first `RUN` fails |
| 28-32 | the `sed` strip list is `dxcam`, `pywin32`, `vgamepad`, `inputs`, `keyboard`. `PyDirectInput` and `mouse` are **not** stripped, contrary to the comment on line 26 — but neither matters, because `game_env.py` imports `win32gui`, `win32process`, `vgamepad` and `dxcam` at module scope, so the environment module is unimportable on Linux regardless (§1) |
| 34 | `pip install -r requirements.txt` still has to resolve `torch==2.5.1+cu121` from PyPI; the `+cu121` index is only consulted on line 35, *after* the failing command inside the same `RUN` |
| 44 | `CMD ["python", "hajime_agent/notebooks/train_agent.py"]` runs with `WORKDIR /app`, but the script inserts `".."` and `"../utils"` into `sys.path` and reads `./demos/`, `./models/`, `file:../mlruns` relative to CWD → `ModuleNotFoundError: game_env` before anything else. Additionally `/app/hajime_agent/notebooks/demos/` is empty, because `.pt` files are git-ignored and so absent from any build context that respects `.gitignore` |

A corrected sketch — **proposal, not tested; no container runtime was exercised for this document**:

```dockerfile
# Proposal only. Still cannot run the environment layer: dxcam/win32gui/vgamepad have no Linux build.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get install -y --no-install-recommends python3.11 python3.11-dev python3.11-venv git ffmpeg \
      libsm6 libxext6 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11
WORKDIR /app/generic_agent/notebooks
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121 \
 && grep -vE '^(dxcam|pywin32|vgamepad|inputs|keyboard|PyDirectInput|mouse)==' /app/requirements.txt > /tmp/req.txt \
 && pip install --no-cache-dir -r /tmp/req.txt
COPY . /app
# Requires a pre-refactored game_env.py with the four Windows imports moved into the non-dummy branch,
# and demonstrations mounted into /app/generic_agent/notebooks/demos.
CMD ["python", "train_agent.py", "--epochs", "10", "--batch", "384", "--lr", "1e-4", "--device", "cuda"]
```

The prerequisite refactor, not the Dockerfile, is the real blocker. Until it is done, the container
path is aspirational and the project should be described as Windows-only.

## 10. Windows-only dependency map

| Package | Purpose in this repository | Linux status | Workaround |
|---|---|---|---|
| `dxcam==0.3.0` | GPU screen capture via Desktop Duplication | Unavailable (WDDM/DXGI only) | `mss` CPU fallback already coded at `game_env.py:141-152`; needs the module-scope import moved inside it |
| `pywin32==311` | `win32gui` window enumeration/rect/move; `win32process` PID lookup | Unavailable | X11/Wayland window queries, or capture a fixed region |
| `vgamepad==0.1.0` | Virtual Xbox 360 pad via ViGEmBus | Windows backend only; `vgamepad.lin` needs `/dev/uinput` | move `import vgamepad` into the non-dummy branch; uinput on Linux |
| `PyDirectInput==1.0.4` | Keyboard/mouse injection in `keyboard_mouse` mode | Unavailable (`ctypes.windll`) | `pynput`/`xdotool`; lazily imported already |
| `keyboard==0.13.5` | Hotkeys (`K`, `L`, `ESC`, arrows, `i/o/p`) | Needs root on Linux | `pynput` |
| `mouse==0.7.1` | Polling mouse-button state while recording | Needs root on Linux | `pynput` |
| `inputs==0.5` | **unused by any repository module** | Cross-platform | drop it |
| `comtypes==1.4.16` | Transitive dependency of `dxcam` | Windows-oriented | goes away with `dxcam` |
| `pygame==2.6.1` | Preview window / event pump in recorder and deploy scripts | Cross-platform | none needed |
| `mss==10.2.0` | CPU capture fallback | Cross-platform | none needed |

## 11. Installation troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `ModuleNotFoundError: No module named 'win32gui'` | `pywin32` post-install step skipped, or a non-Windows host | `python -m pywin32_postinstall -install`; on Linux see §1 and §9 |
| `Exception: VIGEM_ERROR_BUS_NOT_FOUND_ERROR` (or similar `VIGEM_*` name) raised by `import vgamepad` | ViGEmBus driver not installed | Install `ViGEmBus_Setup_x64.exe`, reboot, confirm with `Get-Service ViGEmBus` |
| `ImportError: DLL load failed while importing vgamepad` | 32/64-bit mismatch between the interpreter and the driver library | Use a 64-bit Python 3.11 |
| `pip` reports `No matching distribution found for torch==2.5.1+cu121` | the `+cu121` local labels are absent from PyPI | install the three wheels from `https://download.pytorch.org/whl/cu121` first (§4) |
| `RuntimeError: CUDA error: no kernel image is available` / `torch.cuda.is_available()` is `False` | the pinned `cu121` wheel does not match the installed driver, or the GPU is older than the wheel's supported architectures | pick the index whose CUDA version the driver reports (`nvidia-smi`), or use the CPU wheels and `--device cpu` |
| `python.exe` in `venv\Scripts` fails with "unable to locate base Python" | the committed `venv/` points at `C:\Users\pedro\...\Python311` | delete `venv/` and recreate (§3) |
| `UnicodeDecodeError` / mojibake while loading a `.pt` or writing a report | pickles and generated Markdown were written under a non-UTF-8 default encoding | `set PYTHONUTF8=1` before running; the generated reports are written with `encoding="utf-8"` explicitly |
| `ModuleNotFoundError: game_env` (or `config`) when running a script | wrong working directory | `cd` into `<pkg>/notebooks` (§8) |
| `FileNotFoundError: Directory demos not found!` | fresh clone — demonstrations are git-ignored (`*.pt`) | record some first (§TRAINING_GUIDE 3) |
