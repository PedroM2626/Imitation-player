# Setup

> Getting a clean clone of this repository to a running training job, and an explicit account of
> what does not work. Architecture rationale lives in [ARCHITECTURE.md](ARCHITECTURE.md); the
> operator-facing pipeline lives in [TRAINING_GUIDE.md](TRAINING_GUIDE.md). Every claim below was
> checked against the source in this working copy; where the top-level README disagrees, the
> disagreement is stated.

## 1. Platform reality: the environment layer is Windows-only

`agent/utils/game_env.py` is imported by every CLI entry point. Its module-level imports are portable;
the Windows-only ones are resolved in `agent/utils/windows.py` behind feature flags, so the module
imports on any platform:

| Line | Import | Why it pins the platform |
|---|---|---|
| 26 | `import win32gui` | `pywin32`; window enumeration, `IsWindowVisible`, `GetWindowRect`, `MoveWindow` |
| 27 | `import win32process` | `pywin32`; `GetWindowThreadProcessId` maps a window to a PID |
| game_env.py:239 | `import psutil` | portable, but used only to resolve the PID to a process name |
| 37 | `import vgamepad as vg` | ViGEmBus virtual Xbox 360 pad (`vg.VX360Gamepad()`) |
| 32 | `import dxcam` | Windows Desktop Duplication API (GPU-side screen grab) |

Each of those sits inside `if IS_WINDOWS:` and a `try/except`, setting `HAS_WIN32`, `HAS_DXCAM` and
`HAS_VGAMEPAD` (lines 22-46). `mss` is imported at `windows.py:46` on any platform and used lazily by
`_open_capture`, and `pydirectinput` is imported inside `KeyboardMouseEmitter.__init__`
(`agent/utils/emission.py:139`), so both are optional at import time but still Windows-bound at runtime.
`keyboard` is imported inside the CLI loops that need hotkeys (`agent/cli/record.py`,
`agent/cli/deploy.py`, `agent/cli/dagger.py`), never at module scope; `mouse` is imported lazily inside
the `keyboard_mouse` branch of `agent/utils/input_map.py`.
`test_xinput.py:4` does `ctypes.windll.xinput1_4`, which exists only on Windows.

**Consequence for offline work.** `GenericGameEnv` accepts `config["dummy"] = True` and returns from
`__init__` (line 83) before any window, emitter or camera is created — and because the Windows modules
are now behind the flags in `agent/utils/windows.py`, importing `agent.utils.game_env` on Linux, or on a
Windows machine without the ViGEmBus driver, no longer fails: `vgamepad/__init__.py` still instantiates
its module global `VBUS = VBus()` at import time and `VBus.__init__` raises on any non-`VIGEM_ERROR_NONE`
return, but that failure is caught and turned into `HAS_VGAMEPAD = False`. Dummy-mode training, the
benchmark and the test suite therefore run headless on Linux, which is exactly what CI
(`.github/workflows/ci.yml`) exercises. What remains Windows-only is the live path: recording,
deployment and GAIL construct non-dummy environments, and gamepad mode raises a pointed
`RuntimeError` from `emission.build_emitter` when no pad can be created.

## 2. Hardware

- **GPU.** Any run of practical length needs an NVIDIA GPU with CUDA 12.x. The frozen environment
  pins `torch==2.5.1+cu121` (`requirements.txt:125`).
- **What degrades without CUDA.** `--device` defaults to `None` on every training entry point, and
  `agent/cli/common.resolve_device` (43-51) picks `cuda` when `th.cuda.is_available()` and `cpu`
  otherwise, printing which it chose; an explicit `--device cuda` on a machine with no GPU is honoured and
  raises, as it should. Pass `--device cpu` explicitly on CPU machines and expect the 10-epoch
  benchmark (18 s – 55 min per encoder on GPU, README §10.1) to become impractical.
- **VRAM.** No memory measurement exists anywhere in the repository (no `torch.cuda.max_memory_*`,
  no `nvidia-smi` invocation — verified by grep). The following is arithmetic on the constructor
  defaults, not a measurement. With `--batch 384` over `(4, 128, 128)` float input, one input
  activation is `384 × 4 × 128 × 128 × 4 B ≈ 96 MiB`. The memory hotspot is the Impala-CNN flatten
  head: `ImpalaCNNExtractor` computes `flatten_dim = (128 // 2**3) × (128 // 2**3) × 128 = 16 × 16 ×
  128 = 32 768` (`agent/utils/new_architectures.py:74-76`) and builds `Linear(32768 → 512)` (line 80), i.e.
  16 777 216 weights ≈ 64 MiB of parameters, plus ≈ 64 MiB of gradients and ≈ 128 MiB of Adam
  moments for that single layer, plus a 48 MiB flatten activation per minibatch. The GAP-based
  `ImpoolaCNNExtractor` replaces that input with `AdaptiveAvgPool2d((1,1))`, so its head is
  `Linear(128 → 512)`. Recorded artefact sizes track this: 67.95 MB (Impala-CNN) vs 4.20 MB
  (Impoola-CNN) — README §10.1.
- **Observed parameter counts**, logged as the MLflow metric `num_params` by `agent/cli/train.py:125-141`:
  NatureCNN 4 196 810 · CNN+LSTM+Attention 6 116 779 · ViT 2 448 010 · Impoola-CNN 1 009 258 ·
  Impala-CNN 17 720 938 · ResNet-18 11 516 938 (README §10.1).

## 3. Python

Python **3.11**. Evidence: `requirements.txt` contains only `cp311`-compatible pins, the compiled modules
still present under `agent/utils/__pycache__/` are `*.cpython-311.pyc`, and CI
(`.github/workflows/ci.yml`) pins `python-version: "3.11"`.

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

For everyday work install the curated list instead (§5); README §13.2 does exactly that, and pins
`gymnasium==0.29.1` through `requirements-minimal.txt` rather than restating the versions inline.

## 5. Install path B — curated minimum

`requirements-minimal.txt` is the curated Windows list, derived by grepping every `import` statement in
`agent/` and `test_xinput.py`, with the versions taken from `requirements.txt`;
`requirements-docker.txt` is the same list for Linux/CUDA training, minus the Windows-only capture and
actuation packages (`dxcam`, `pywin32`, `vgamepad`, `keyboard`, `mouse`, `PyDirectInput`, `inputs`,
`comtypes`).

```powershell
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-minimal.txt
pip install pytest
```

On Linux, the CPU wheels plus the container list are what CI installs:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-docker.txt
```

| Package | Imported by |
|---|---|
| `torch`, `torchvision` | every `agent/utils/*.py`; `torchvision.models.resnet18` in `agent/utils/new_architectures.py:11` |
| `gymnasium`, `stable-baselines3` | environment and policy classes |
| `imitation` | `imitation.algorithms.BC`, `imitation.data.types.Trajectory`, `imitation.GAIL` |
| `numpy`, `opencv-python` | array handling and `cv2.resize` / colour conversion |
| `mlflow` | explicit import in `agent/utils/tracking.py`; **not** a declared dependency of `imitation==1.0.1`, so it must be installed by hand |
| `psutil`, `pywin32`, `vgamepad`, `dxcam` | the Windows capture/actuation layer, behind `agent/utils/windows.py` |
| `keyboard`, `pygame` | hotkeys and the preview window in `agent/cli/record.py`, `agent/cli/deploy.py` and `agent/cli/dagger.py` |
| `mss`, `PyDirectInput`, `mouse` | lazily: `mss` on DXCam failure, the other two in `keyboard_mouse` mode |
| `comtypes` | dependency of `dxcam`; imported by no repository module, and not pinned by either curated list — only by the freeze |

`inputs==0.5` appears in `requirements.txt:55` but is imported by **nothing** in the tree (verified by
grep) — it is dead weight in the frozen file, and both curated lists leave it out.

## 6. ViGEmBus

`vgamepad` is a Python binding for the **ViGEmBus** kernel driver; the pip package does not install the
driver. Install it from the ViGEm repository (`ViGEmBus_Setup_x64.exe`) and reboot.

Two failure modes, in order of severity:

1. **Driver absent → `import vgamepad` raises** at module load — but that import now sits inside the
   `try/except` in `agent/utils/windows.py:38-42`, so it is swallowed into `HAS_VGAMEPAD = False` instead
   of breaking the whole package. The failure surfaces later and only where it matters: constructing a
   live gamepad-mode environment raises `RuntimeError("input_mode='gamepad' needs vgamepad and the
   ViGEmBus driver …")` from `emission.build_emitter`. Dummy-mode training is unaffected. The underlying
   traceback still mentions a `VIGEM_ERRORS` name (`VIGEM_ERROR_BUS_NOT_FOUND_ERROR` in practice).
2. **Driver present but the game ignores the pad** — some titles poll only DirectInput, or were started
   before the virtual device was attached. Restart the title after the pad exists.

The README's §13.3 claim that "the virtual pad is created but no game receives input, and the failure
is silent" understates case 1: with the driver missing, `import vgamepad` does not silently succeed — it
raises, and `agent/utils/windows.py` records the absence as `HAS_VGAMEPAD = False`.
For the record, `Get-Service ViGEmBus` reports the service **absent** on the machine this document was
written on, so no deploy or recording command in this repository can currently be executed there; offline
training is unaffected (§1).

## 7. Game and emulator prerequisites

- **Start the title before constructing the environment.** `find_window_by_process_name`
  (`agent/utils/game_env.py:236-255`) enumerates visible top-level windows and matches the owning process name
  case-insensitively against `GAME_CONFIG["process_name"]`, returning the **first** hit in enumeration
  order. If no window is found and `exe_path` is set, `__init__` does `Popen([exe_path, rom_path])`
  (`agent/utils/game_env.py:87-92`), then `wait_start()` polls for one second at a time for **up to 120 seconds**
  (`agent/utils/game_env.py:257-275`). On timeout it **raises `WindowNotFoundError`** with a message naming
  the process and the timeout, instead of printing a warning and leaving an object whose every later
  capture yields the black/previous-frame fallback; the CLI wrappers print that message and exit cleanly
  (`agent.cli.common.cli_entry`). **Look for the `Found window HWND:` line before trusting a recording.**
- **Window offsets.** The grab rectangle is the window rect plus `window_offset.left/top` and minus
  `window_offset.right/bottom` (`_window_region`, `agent/utils/game_env.py:109-116`). The committed defaults
  (`left: 20, top: 100`) exist to skip an emulator's title bar and border. Adjust them per machine, or
  run the game borderless/fullscreen and zero them.
- **Machine-specific paths are not in the profiles.** Both `agent/config/profiles/hajime_ippo.py` and
  `agent/config/profiles/roblox.py` set `"exe_path": None` and `"rom_path": None`; the emulator executable
  and ROM that used to be hard-coded in the Hajime config are supplied instead in
  `agent/config/local.py`, copied from `agent/config/local.example.py` and git-ignored. The loader
  (`agent/config/__init__.py:47-59`) merges `LOCAL_OVERRIDES[<profile>]` over the profile and honours only
  `exe_path`, `rom_path` and `process_name`. Leaving `exe_path` as `None` means the target window must
  already be open.
- **`keyboard_mouse` mode** (the `roblox` profile) additionally requires
  `actions.input_mode == "keyboard_mouse"` and a mapping table whose entries use `kind` (`key` or
  `mouse_button`) with `key`/`button` — the same table and the same `kind` schema the gamepad profile
  uses, now that both input modes drive emission
  ([TRAINING_GUIDE §2](TRAINING_GUIDE.md#2-editing-agentconfigprofileshajime_ippopy)).

## 8. Verification, in order

Every entry point is a module invocation from the **repository root**, and every path it uses is derived
from the profile (`agent/utils/paths.py`), so no command below depends on the working directory beyond
being run where `agent/` is importable. `--profile` and `--runs-root` (or `IMITATION_PROFILE` /
`IMITATION_RUNS`) redirect what is read and written. The quickest whole-tree check is
`python -m pytest -q` — 69 tests, CPU only, no game and no driver required — and it is exactly what
`.github/workflows/ci.yml` runs on `ubuntu-latest`.

| # | Command (PowerShell) | Expected observable outcome |
|---|---|---|
| 1 | `python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"` | Python 3.11.x, `2.5.1+cu121`, `True`, `12.1`. Confirms the venv is active, the CUDA wheel resolved, and the driver matches the wheel |
| 2 | `python -c "from agent.config import load_profile, num_actions; g=load_profile('hajime_ippo')['GAME_CONFIG']; print(num_actions(g), g['actions'].get('input_mode'))"`<br>`python -c "from agent.config import load_profile, num_actions; g=load_profile('roblox')['GAME_CONFIG']; print(num_actions(g), g['actions'].get('input_mode'))"` | `18 gamepad` for `hajime_ippo`; `9 keyboard_mouse` for `roblox` — both profiles now declare `input_mode` explicitly. Proves the profile loader resolves and that `num_actions` agrees with each profile's mapping table; `agent.config.num_actions` raises `ValueError` if the two disagree |
| 3 | `python -c "from agent.utils.game_env import GenericGameEnv; from agent.config import load_profile; cfg=dict(load_profile('hajime_ippo')['GAME_CONFIG']); cfg['dummy']=True; e=GenericGameEnv(cfg); print(e.observation_space, e.action_space)"` | a `Box` of shape `(128, 128, 1)` dtype `uint8` and `MultiBinary(18)` (`MultiBinary(9)` under `roblox`). This succeeds **without Windows or ViGEmBus**, because the platform imports sit behind `agent/utils/windows.py`; on a Windows box with the driver it is also an indirect test that `import vgamepad` connected |
| 4 | `python -c "from agent.cli.common import build_config, wrapped_env; from agent.config import load_profile; cfg=build_config(load_profile('hajime_ippo')['GAME_CONFIG'], dummy=True); v=wrapped_env(cfg); print(v.observation_space, v.action_space)"` | `Box(0, 255, (4, 128, 128), uint8)`. This is the exact wrapper stack every script now builds (`agent/cli/common.wrapped_env`), i.e. the space every encoder is constructed against |
| 5 | `python test_xinput.py` | four lines, `Controller i: res=..., buttons=..., LX=..., LY=...`. Observed on the reference machine with no pad attached: `res=1167` (`ERROR_DEVICE_NOT_CONNECTED`) on all four slots. Slot *N* returning `res=0` after a training run started means the virtual pad exists. **This probe does not exercise ViGEmBus by itself** — it only reads XInput slots, so `res=1167` everywhere is not evidence of a missing driver |
| 6 | `python -m agent.cli.train --profile hajime_ippo --arch naturecnn --epochs 1 --batch 32 --lr 1e-4 --device cpu` | `BEHAVIOURAL CLONING - NatureCNN`, the profile/epochs/batch/lr/device line, a corpus summary (trajectories, frames, distinct joint actions, modal and top-5 shares, the marginal baseline, the per-bit firing table), one epoch of progress, then `[OK] runs/hajime_ippo/models/bc_policy.zip` and the loss against the baseline. On an **empty** demos directory it exits with `no files matching 'demo*.pt' in … Record some first`, which is the correct outcome for a fresh clone |

Step 6 needs a note about the shipped corpus: `runs/hajime_ippo/demos/` holds two 7-wide files, and
`hajime_ippo` declares `num_actions = 18`, so under the default `strict` width policy the command above
stops with `DemoError: demo_0_20260523_224241.pt: actions have width 7 but the profile declares
num_actions=18`. That refusal is the intended behaviour (§DATA 6.1); pass `--width-policy coerce` to
reproduce the truncated/padded reconciliation the published benchmark used, which prints a warning per
file first.

Step 6 writes into `runs/hajime_ippo/models/`, `runs/hajime_ippo/logs/` and `runs/hajime_ippo/mlruns/`;
delete `runs/hajime_ippo/models/bc_policy.zip` afterwards if you intend to benchmark, since
`agent.cli.deploy` prefers that file over any numbered checkpoint
([TRAINING_GUIDE §8](TRAINING_GUIDE.md#8-deploying-a-policy)).

## 9. Docker: the `Dockerfile` is repaired but unverified

The file installs a toolchain and a Linux-installable dependency list, and its default command is a
training smoke run. It is still **untested**: it has never been built, and no container runtime was
exercised for this document. Contents, in build order:

| Line | What it does |
|---|---|
| 12 | Base is `nvidia/cuda:12.1.1-runtime-ubuntu22.04` — the **runtime** flavour, which ships the CUDA runtime libraries but no Python toolchain; the next layer supplies one |
| 21-25 | `apt-get install -y --no-install-recommends python3.11 python3.11-venv python3-pip libgl1 libglib2.0-0`, then points `/usr/bin/python` at 3.11 with `update-alternatives`. The two `libgl1`/`libglib2.0-0` packages are the OpenCV runtime libraries the frozen list used to assume |
| 29-33 | installs `torch==2.5.1+cu121` and `torchvision==0.20.1+cu121` from the PyTorch CUDA index **first**, then the rest from `requirements-docker.txt` — the curated Linux list, not the freeze, so no `sed` strip of `dxcam`/`pywin32`/`vgamepad`/`inputs`/`keyboard` is needed and nothing is left to strip inconsistently |
| 35-37 | `COPY agent`, `COPY runs`, `COPY README.md`. Because `runs/` is copied wholesale, the ~3.6 GB of `.pt` demonstrations, the `models/` checkpoints and `mlruns/` are baked into the image; the repository still has **no `.dockerignore`** |
| 40 | `CMD ["python", "-m", "agent.cli.train", "--arch", "impoola", "--epochs", "1", "--batch", "64", "--device", "cpu"]` — a one-epoch CPU smoke run, so `docker run` exercises the install end to end rather than crashing on a missing interpreter |

```dockerfile
# Actual Dockerfile, abridged. Offline BC works here because the environment is
# created with dummy=True; recording, deployment and GAIL do not — the capture and
# actuation layer is Windows-only by nature.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV IMITATION_PROFILE=hajime_ippo
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip libgl1 libglib2.0-0 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
WORKDIR /app
COPY requirements-docker.txt /app/requirements-docker.txt
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    && python -m pip install -r requirements-docker.txt
COPY agent /app/agent
COPY runs /app/runs
CMD ["python", "-m", "agent.cli.train", "--arch", "impoola", "--epochs", "1", "--batch", "64", "--device", "cpu"]
```

The refactor that used to block this path — moving the four Windows imports out of `game_env.py`'s module
scope — has been done (`agent/utils/windows.py`, §1), and the same offline training runs on Linux in CI
without a container. What remains unproven is the image itself. Recording, deployment and GAIL still
cannot run in it: the capture and actuation layer is Windows-only by nature, so the project remains
Windows-only for the live path and cross-platform for the offline one.

## 10. Windows-only dependency map

| Package | Purpose in this repository | Linux status | Workaround |
|---|---|---|---|
| `dxcam==0.3.0` | GPU screen capture via Desktop Duplication | Unavailable (WDDM/DXGI only) | imported behind `windows.HAS_DXCAM`; the `mss` CPU fallback is coded at `agent/utils/game_env.py:133-138`, and the module now imports with neither present |
| `pywin32==311` | `win32gui` window enumeration/rect/move; `win32process` PID lookup | Unavailable | `windows.HAS_WIN32` gates it; window lookup simply finds nothing off Windows |
| `vgamepad==0.1.0` | Virtual Xbox 360 pad via ViGEmBus | Windows backend only; `vgamepad.lin` needs `/dev/uinput` | `windows.HAS_VGAMEPAD` gates the import; gamepad mode then raises a pointed error from `emission.build_emitter`. uinput on Linux is still the missing piece |
| `PyDirectInput==1.0.4` | Keyboard/mouse injection in `keyboard_mouse` mode | Unavailable (`ctypes.windll`) | `pynput`/`xdotool`; lazily imported already (`agent/utils/emission.py:139`) |
| `keyboard==0.13.5` | Hotkeys (`K`, `L`, `ESC`) and the keyboard stand-ins for every mapping (`↑ ↓ ← →`, `i/o/p/u`, `j/k`, `;`) | Needs root on Linux | `pynput`; imported inside the CLI loops, so the package imports without it |
| `mouse==0.7.1` | Polling mouse-button state while recording | Needs root on Linux | `pynput` |
| `inputs==0.5` | **unused by any repository module** | Cross-platform | dropped from both curated lists; still in the freeze |
| `comtypes==1.4.16` | Transitive dependency of `dxcam` | Windows-oriented | goes away with `dxcam`; excluded from `requirements-docker.txt` |
| `pygame==2.6.1` | Preview window / event pump in the recorder and the DAgger collection loop | Cross-platform | none needed |
| `mss==10.2.0` | CPU capture fallback | Cross-platform | none needed |

## 11. Installation troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `ModuleNotFoundError: No module named 'win32gui'` | `pywin32` post-install step skipped | `python -m pywin32_postinstall -install`; on Linux the import is gated by `agent/utils/windows.py`, so this only costs live capture (§1 and §9) |
| `Exception: VIGEM_ERROR_BUS_NOT_FOUND_ERROR` (or similar `VIGEM_*` name) raised by `import vgamepad` | ViGEmBus driver not installed | Install `ViGEmBus_Setup_x64.exe`, reboot, confirm with `Get-Service ViGEmBus`. Off Windows the same failure is swallowed into `HAS_VGAMEPAD = False` (§6) |
| `ImportError: DLL load failed while importing vgamepad` | 32/64-bit mismatch between the interpreter and the driver library | Use a 64-bit Python 3.11 |
| `pip` reports `No matching distribution found for torch==2.5.1+cu121` | the `+cu121` local labels are absent from PyPI | install the three wheels from `https://download.pytorch.org/whl/cu121` first (§4) |
| `RuntimeError: CUDA error: no kernel image is available` / `torch.cuda.is_available()` is `False` | the pinned `cu121` wheel does not match the installed driver, or the GPU is older than the wheel's supported architectures | pick the index whose CUDA version the driver reports (`nvidia-smi`), or use the CPU wheels and `--device cpu` |
| `python.exe` in `venv\Scripts` fails with "unable to locate base Python" | the stale `venv/` points at `C:\Users\pedro\...\Python311` | delete `venv/` and recreate (§3) |
| `UnicodeDecodeError` / mojibake while loading a `.pt` or writing a report | pickles and generated Markdown were written under a non-UTF-8 default encoding | `set PYTHONUTF8=1` before running; the generated reports are written with `encoding="utf-8"` explicitly |
| `ModuleNotFoundError: No module named 'agent'` | run from a directory that is not the repository root, or with a per-script path layout that no longer exists | `python -m agent.cli.<name>` from the repository root (§8) |
| `DemoError: actions have width N but the profile declares num_actions=M`, or `DemoError: K demonstration file(s) could not be read` | the corpus mixes action widths under the default `strict` `width_policy`, or a `.pt` file is corrupt — every failure is now listed rather than skipped | re-record, pick the matching `--profile`, or pass `--width-policy coerce` to truncate/pad with a warning; delete or re-export unreadable files. Demonstrations are git-ignored (`*.pt`), so on a fresh clone record some first ([TRAINING_GUIDE §3](TRAINING_GUIDE.md#3-recording-demonstrations)) |
