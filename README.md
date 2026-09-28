# Imitation Player

**A player-agnostic imitation-learning stack that learns a visuomotor policy from raw screen pixels.**

*Imitation Player* records a human playing any PC game or console emulator, and learns to map
`screenshot → gamepad/keyboard action` by supervised behavioural cloning. The resulting policy is
then executed back into the game through a **virtual** controller, closing a perception–action loop
in which the agent never has access to the game's internal state — only to what is rendered on screen.

The project is organised as a research codebase rather than a packaged library: a screen-capture
Gymnasium environment, a set of interchangeable CNN / recurrent / attention feature extractors, and
command-line pipelines for demonstration recording, training, benchmarking and interactive replay.

---

## Table of contents

| Section | Contents |
|---|---|
| [1. Motivation and scope](#1-motivation-and-scope) | Problem statement, what is and is not claimed |
| [2. System overview](#2-system-overview) | Data flow, pipeline stages |
| [3. Repository layout](#3-repository-layout) | Package structure, configuration profiles and the `runs/` data root |
| [4. Related work and positioning](#4-related-work-and-positioning) | How this sits relative to the literature |
| [5. Environment formalisation](#5-environment-formalisation) | Observation space, action space, reward, termination |
| [6. Demonstration corpus](#6-demonstration-corpus) | Recording protocol and measured dataset statistics |
| [7. Policy architectures](#7-policy-architectures) | The six benchmarked encoders, layer by layer |
| [8. Training objectives and algorithms](#8-training-objectives-and-algorithms) | BC, DAgger, GAIL |
| [9. Experimental protocol](#9-experimental-protocol) | Hyperparameters, budget, instrumentation |
| [10. Results](#10-results) | Benchmark table, repeated-run dispersion, trivial baselines |
| [11. Analysis](#11-analysis) | Parameter efficiency vs. fit, architecture comparison |
| [12. Threats to validity](#12-threats-to-validity) | Limitations of the current evidence |
| [13. Getting started](#13-getting-started) | Installation and first run |
| [14. Usage guide](#14-usage-guide) | Step-by-step pipeline operation |
| [15. Reproducibility](#15-reproducibility) | Seeds, artefacts, what is and is not versioned |
| [16. Troubleshooting](#16-troubleshooting) | Known failure modes |
| [17. Documentation index](#17-documentation-index) | Deep-dive documents |
| [18. License and attribution](#18-license-and-attribution) | Credits |

---

## 1. Motivation and scope

Learning to play a game from pixels is usually formulated as reinforcement learning, which requires
either a hand-written reward function or a dense task signal. Neither exists for an arbitrary
off-the-shelf game running in an emulator. Behavioural cloning (BC) sidesteps this: the reward is
implicit in the demonstrations. This project investigates how far a **fully black-box** version of
that idea can be taken — the agent observes only the framebuffer and actuates only a virtual
controller, with no hooks into the game process, no memory reading, and no engine integration.

### What this repository does provide

- A **screen-capture Gymnasium environment** (`GenericGameEnv`) that finds a window by process name,
  grabs it through the GPU desktop-duplication API, downsamples it to a fixed resolution, and emits
  actions through a virtual XInput gamepad or synthesised keyboard/mouse events.
- A **demonstration recorder** that captures synchronised `(observation, action)` pairs while a human
  plays, and serialises them in the `imitation` library's `Trajectory` format.
- **Six interchangeable feature extractors** (CNN, recurrent-with-attention, transformer, residual
  with global average pooling, residual with flatten, and a pretrained-family deep residual net)
  behind a single training entry point.
- An **automated architecture benchmark** (`agent.cli.benchmark`) that trains each encoder under an
  identical budget and writes the comparison into this README and into
  `runs/hajime_ippo/models/comparison_results.md`.
- Pipelines for **BC**, **DAgger** (interactive correction) and **GAIL** (adversarial imitation).
- **MLflow** instrumentation of loss, gradient norm, entropy and action-likelihood probability.

### What this repository does *not* (yet) provide

Stated explicitly, because it bounds how the numbers in §10 may be read:

- **No held-out evaluation.** There is no train/validation split, no per-action accuracy, no
  closed-loop rollout metric and no success-rate measure anywhere in the code base. Every figure in
  §10 is an **in-sample training loss**. See §12.
- **No task-level evidence that the agent can play.** The policies were never scored against the
  game; the qualitative claim "the AI plays the game" is untested.
- **No repeated-run statistics in the headline table.** Six architectures were compared with 1–4
  runs each; dispersion between runs of the *same* architecture (§10.2) is of the same order as the
  differences the table is used to rank.
- **No reference baseline, until now — and it is unflattering.** A predictor that ignores the screen
  entirely and merely reproduces each action bit's empirical marginal attains 2.6066 nats on the labels
  this benchmark uses. Every architecture in §10.1 scores *worse* than that (2.90 – 3.48). See §10.3.
- **No benchmark against external baselines** (e.g. the original CNN+LSTM agents of the literature).

---

## 2. System overview

```
     PHASE 1            PHASE 2                    PHASE 3              PHASE 4 (optional)
   RECORD             BEHAVIOURAL CLONING         DEPLOY               DAGGER REFINEMENT
      |                       |                        |                       |
      v                       v                        v                       v
 +-----------+        +------------------+      +----------------+     +------------------+
| Human plays |  -->  |  Supervised fit  | -->  |  Policy drives | --> | Human overrides  |
| game; screen|        | screen -> action |      |  virtual       |     | the agent; the   |
| + inputs    |        | (CNN/RNN/ViT)    |      |  gamepad       |     | corrections are  |
| -> .pt demo |        | MLflow tracked   |      |                |     | appended -> retrain
 +-----------+        +------------------+      +----------------+     +------------------+
                              ^                                                |
                              +------------- aggregated dataset <--------------+

     ALTERNATIVE (Phase 5): GAIL — an adversarial discriminator replaces the
     supervised objective, and the generator improves against the learned
     reward. Requires the live game, i.e. the environment cannot be in dummy mode.
```

| Stage | Script | Input | Output |
|---|---|---|---|
| Record | `agent/cli/record.py` | live game + human input | `runs/<profile>/demos/demo_*.pt` |
| Clone | `agent/cli/train.py --arch <name>` | `demos/*.pt` | `runs/<profile>/models/*.zip`, MLflow run |
| Benchmark | `agent/cli/benchmark.py` | `demos/*.pt` | 6 policies + generated tables |
| Deploy | `agent/cli/deploy.py --arch <name>` | `models/*.zip` | live play through virtual pad |
| Correct | `agent/cli/dagger.py` | live game + policy + human | additional demos, then retrains |
| Adversarial | `agent/cli/train_gail.py` | live game + demos | adversarially trained policy |

---

## 3. Repository layout

One package, one data root, and per-title configuration profiles. (Earlier revisions shipped the code
twice, as `hajime_agent/` and `generic_agent/`, with ten of eighteen files byte-identical between them;
that duplication was removed by collapsing to `agent/` plus `config/profiles/`.)

```
imitation-player/
│
├── agent/
│   ├── config/
│   │   ├── __init__.py           # Profile loader, local-override merge, width validation
│   │   ├── local.example.py      # Machine-specific exe/ROM paths (copy to local.py, git-ignored)
│   │   └── profiles/
│   │       ├── hajime_ippo.py    # rpcs3 + gamepad + 18 actions (the corpus that exists today)
│   │       └── roblox.py         # keyboard/mouse + 9 actions (example profile, never trained)
│   ├── utils/
│   │   ├── game_env.py           # GenericGameEnv: capture, actuation, pacing, window discovery
│   │   ├── windows.py            # Optional-import feature flags (win32 / dxcam / vgamepad / mss)
│   │   ├── emission.py           # Mapping-driven action output (gamepad or keyboard/mouse)
│   │   ├── input_map.py          # Human-input capture: XInput, keyboard, mouse -> action bits
│   │   ├── demos.py              # Loading, integrity policy, corpus statistics, trivial baselines
│   │   ├── tracking.py           # MLflow sink, metric capture, per-run dataset provenance
│   │   ├── architectures.py      # Registry of the six encoders: names, kwargs, checkpoint prefixes
│   │   ├── paths.py              # runs/<profile>/{demos,models,mlruns,logs}
│   │   ├── checkpoints.py        # Numeric checkpoint discovery and resolution
│   │   ├── utils.py              # PolicyRunner, load_policy
│   │   ├── temporal_lstm.py      # CNN + bidirectional LSTM + temporal attention extractor
│   │   ├── impoola_cnn.py        # Impoola-CNN (GAP) extractor
│   │   ├── new_architectures.py  # Impala-CNN (Flatten) and ResNet-18 extractors
│   │   └── vision_transformer.py # ViT extractor with spatial and temporal embeddings
│   └── cli/
│       ├── common.py             # Shared argument parsing, device resolution, env wrapping
│       ├── record.py             # Demonstration capture
│       ├── train.py              # Behavioural cloning, --arch selects the encoder
│       ├── benchmark.py          # Automated architecture study + generated report
│       ├── deploy.py             # Play the game with a trained policy
│       ├── dagger.py             # Interactive DAgger correction collection
│       └── train_gail.py         # GAIL (adversarial imitation) -- experimental, never completed
│
├── runs/
│   └── hajime_ippo/
│       ├── demos/*.pt            # Recorded trajectories (not versioned)
│       ├── models/
│       │   ├── *.zip             # Trained policies (not versioned; see §15.2)
│       │   └── comparison_results.md   # Generated by agent.cli.benchmark
│       ├── logs/progress.csv     # Retained historical BC training log
│       └── mlruns/               # Local MLflow store (not versioned)
│
├── tests/                        # pytest suite, run on Linux in CI
├── .github/workflows/ci.yml
├── docs/                         # Deep-dive documentation (see §17)
├── test_xinput.py                # Standalone XInput controller probe
├── requirements.txt              # Verbatim pip freeze of the workstation that produced the results
├── requirements-minimal.txt      # Curated Windows install list
├── requirements-docker.txt       # Curated Linux/CUDA install list (no Windows-only packages)
├── pytest.ini
├── Dockerfile                    # CUDA training image; untested build, §13.4
├── LICENSE                       # MIT
├── .gitattributes                # Git LFS rules for model archives
└── .gitignore
```

Everything is invoked from the repository root as a module:

```bash
python -m agent.cli.train --profile hajime_ippo --arch impoola --epochs 100
```

The scripts no longer depend on the current working directory: paths are derived from the profile by
`agent/utils/paths.py`, so `cd`-ing into a package folder is unnecessary (the old scripts had to be run
from `<pkg>/notebooks`). `IMITATION_PROFILE` sets the default profile and `IMITATION_RUNS` relocates the
data root; every CLI also accepts `--profile` and `--runs-root`.

### 3.1 Profiles instead of packages

The two former packages are now two profiles, and the only real differences are the ones a
configuration should carry:

| Aspect | `hajime_ippo` | `roblox` |
|---|---|---|
| `process_name` | `rpcs3` | `RobloxPlayerBeta` |
| `actions.input_mode` | `gamepad` (virtual Xbox 360 pad) | `keyboard_mouse` (synthesised events) |
| `actions.num_actions` | 18 | 9 |
| `deploy.fps` | 30 | 60 |
| Emulator / ROM paths | `None` in the profile; supplied via `config/local.py` | `None` |

Everything else — the environment, the emitters, the recorders, the extractors, the trainer — is a
single implementation shared by both, which means a fix applied once now applies to every title. The
corpus and all recorded benchmark evidence belong to the `hajime_ippo` profile (§6.3): the `roblox`
profile exists as a worked example of configuring another game and has never been trained on.

---

## 4. Related work and positioning

The design is a direct descendant of screen-based deep imitation work rather than of API-integrated
game agents.

- **DAC / Simulation Mining** (Baker et al., 2019) — learning a controller from video
  demonstrations with synthetic augmentation. *Imitation Player* shares the black-box pixel premise
  but omits their action-space alignment machinery.
- **Video PreTraining (VPT)** (Sidorov et al., 2021) — six scales of Minecraft demonstration data,
  with an inverse-dynamics model labelling mouse/keyboard actions from frames. Our recorder avoids
  the inverse-dynamics step entirely by tapping the real input device.
- **IMPALA / CRN** (Espeholt et al., 2018) — the convolutional "simple" recurrent network that
  motivates both the Impala-CNN baseline and our Impoola-CNN variant (§7.4, §7.5).
- **Nature-CNN** (Mnih et al., 2015) — the three-layer encoder adopted verbatim by
  Stable-Baselines3 as the default `CnnPolicy`, used here as the cheap reference point (§7.1).
- **DAgger** (Ross, Gordon & Bagnell, 2011) — aggregation of expert corrections along the learner's
  own state distribution; implemented as an interactive collection loop (§8.2).
- **GAIL** (Ho & Erlikhman, 2016) — adversarial imitation where a discriminator supplies the
  reward; implemented but never completed (§8.3, §12.5).
- **ViT** (Dosovitskiy et al., 2021) and **ResNet** (He et al., 2016) — imported as
  computer-vision backbones and adapted to a 4-channel stacked-frame input (§7.3, §7.6).
- **Swin Transformer** (Liu et al., 2021) and **ConvNeXt** (Liu et al., 2022) — evaluated, then
  withdrawn for cost reasons; the withdrawn attempts are documented in §7.7.
- **Mamba / state-space models** — one exploratory `Vision_Mamba` experiment series exists in the
  tracking store but its source file no longer exists in the tree (§7.7).

Methodologically the project sits at the *engineering/demonstration* end of the literature: it
reproduces a known recipe (pixels-to-actions BC over a virtual pad) and uses it as a harness for an
encoder comparison, rather than proposing a new objective.

---

## 5. Environment formalisation

`GenericGameEnv` (`agent/utils/game_env.py`) subclasses `gymnasium.Env`. It is best read as a
**partially specified MDP**: the state and action channels are fully implemented, while reward and
termination are deliberately degenerate because BC does not consume them.

### 5.1 Observation space

```python
Box(low=0, high=255, shape=(128, 128, 1), dtype=uint8)   # single HWC grayscale frame
```

The model, however, never sees that space directly. The training scripts wrap the environment in
`DummyVecEnv -> VecTransposeImage -> VecFrameStack(n_stack=4)` (`agent/utils/paths`-rooted, built by `agent/cli/common.wrapped_env`), so the
tensor that reaches a feature extractor is `(B, 4, 128, 128)` in **CHW** order with `C` = the four
most recent frames. Recorded demonstrations are already stored as `(N, 4, 128, 128)`, i.e. stacked
at capture time by the recorder.

Two consequences, both of which matter when reading §7:

1. Extractors that infer `n_frames = observation_space.shape[0]` are correct only on the **wrapped**
   space; they cannot be constructed from the raw environment space.
2. The recurrent encoder (§7.2) additionally buffers 10 CNN outputs internally, so the effective
   temporal receptive field is 4 stacked frames × 10 buffered features, and the two mechanisms are
   redundant rather than complementary.

**Capture path.** The target window is located by enumerating visible top-level windows and matching
the owning process name (`win32gui.EnumWindows` + `psutil`). The grab region is the window rectangle
minus configurable offsets. `dxcam` (Windows Desktop Duplication, GPU side, grayscale output) is the
primary path; `mss` (CPU) is a fallback that engages on any initialisation exception — typically on
dual-GPU laptops where duplication attaches to the wrong adapter. Frames are resampled to 128×128
with `cv2.INTER_NEAREST`. If a grab returns nothing, the previous frame is re-emitted, and a black
frame is emitted when no previous frame exists; this is a **silent** degradation path (§12.3).

### 5.2 Action space

```python
MultiBinary(num_actions)     # num_actions = 18 (hajime_ippo profile) or 9 (roblox profile)
```

The 18-dimensional gamepad layout, as implemented in `step()` (indices are hard-coded):

| Index | Semantic | Emitted as | Magnitude |
|---|---|---|---|
| 0 | D-pad / left stick up | left stick `y = -1.0` | boolean |
| 1 | down | left stick `y = +1.0` | boolean |
| 2 | left | left stick `x = -1.0` | boolean |
| 3 | right | left stick `x = +1.0` | boolean |
| 4 | face button A (Cross) | `XUSB_GAMEPAD_A` | boolean |
| 5 | face button B (Circle) | `XUSB_GAMEPAD_B` | boolean |
| 6 | face button X (Square) | `XUSB_GAMEPAD_X` | boolean |
| 7 | left trigger | `left_trigger(255)` | boolean |
| 8 | right trigger | `right_trigger(255)` | boolean |
| 9 | left stick press | `XUSB_GAMEPAD_LEFT_THUMB` | boolean |
| 10–13 | right stick x (camera) | `+0.5, +1.0, −0.5, −1.0` | 4-way discretised |
| 14–17 | right stick y (camera) | `−0.5, −1.0, +0.5, +1.0` | 4-way discretised |

Axis directions are encoded as **orthogonal binary flags** rather than a continuous value, so a
single network head (`MultiBinary` logits) can express them; conflicting flags within an axis group
resolve to whichever index is tested last (`game_env.py:248-261`), an implicit and undocumented
precedence.

The 9-dimensional keyboard/mouse layout used by the `roblox` example profile maps each
index to a `pydirectinput` key or mouse button through the `actions.mappings` table.

> **How the table is used.** In both input modes `actions.mappings` is the single source of truth:
> `agent/utils/emission.py` validates it against `actions.num_actions` at construction and drives the
> virtual device from it, and `agent/utils/input_map.py` reads the human's physical controls back
> through the same table, so a recording and a replay agree on what every bit means. Axis bits **sum and
> clamp** rather than letting one index override another. (The previous implementation hard-coded index
> → button for gamepad mode and ignored the table entirely; conflicting flags resolved to whichever index
> was tested last.)

### 5.3 Reward, termination, and the deployment loop

`step()` returns `observation, 0.0, False, False, {}`. Reward is constant zero and neither
`terminated` nor `truncated` ever becomes `True`; `reset()` clears only the pressed-key set. This is
sound for BC, which ignores both channels, but it means the environment as written **cannot support
any episodic evaluation metric** — there is no notion of a round ending, a life being lost, or a
match being won. Any future success-rate work requires a reward/terminal signal, which in a
black-box setting must be inferred from pixels (a HUD reader) or from the game's own state.

Pacing is implemented rather than merely declared: `GenericGameEnv.step()` sleeps out the remainder of the
`1 / capture.target_fps` frame budget, and `agent.cli.deploy` additionally caps its own loop at
`deploy.fps` (30 for `hajime_ippo`, 60 for `roblox`). The previous revision computed a `frame_time` it
never used and carried a `MAX_FPS` constant that nothing read, so deployment ran as fast as capture and
inference allow. Because capture is *asynchronous* to the emulator's own
frame rate, the recorded `(frame, action)` alignment is only as tight as the operator's reaction
time, and the same action may be logged across several consecutive frames.

### 5.4 Dummy mode

Setting `config["dummy"] = True` returns from `__init__` before window lookup, gamepad creation and
camera acquisition. This allows offline training on recorded demonstrations without a running game
and is what every training script uses. It is **not** a functioning offline environment: `step()`
still dereferences `self.gamepad`, so any code path that steps a dummy environment raises. GAIL,
which must generate fresh trajectories, therefore forces `dummy = False` and cannot run headless.

### 5.5 Configuration surface: which control does what

An earlier revision of this document listed eleven configuration fields that were read by nothing —
knobs that looked tunable and were decorative, including `capture.width`, `TRAINING_CONFIG.window_size`,
`dagger_iterations`, `demo_path`, and the whole of `INPUT_CONFIG`. That has been remediated: every field
below is consumed, and the two that never had a coherent meaning were deleted.

| Control | Consumer | Effect |
|---|---|---|
| `process_name` | `game_env.find_window_by_process_name` | case-insensitive **substring** of the owning process name; first matching window in enumeration order wins |
| `exe_path`, `rom_path` | `game_env.__init__` | used only if no window exists: `Popen([exe_path, rom_path])`. Both are `None` in a profile and come from `config/local.py` |
| `hide_window` | `game_env.wait_start` | moves the window off-screen; the capture region follows it |
| `capture.internal_width/height` | observation space, `cv2.resize`, every extractor | model input resolution; also sets the Impala-CNN flatten width, which needs divisibility by 8 (asserted in `tests/test_config.py`) |
| `capture.target_fps` | `dxcam.start`, and `GenericGameEnv.step` pacing | capture rate **and** the per-step frame budget the environment now sleeps out |
| `capture.buffer_len` | `dxcam.create(max_buffer_len=…)` | `1` is the lowest-latency setting |
| `window_offset.*` | `_window_region` | compensates title bars and emulator borders |
| `actions.num_actions` | action space, recorder width, demo validation | must equal `len(actions.mappings)` or profile loading raises |
| `actions.input_mode` | `build_emitter`, `HumanInput` | `gamepad` or `keyboard_mouse` |
| `actions.mappings` | emission **and** human capture | single source of truth for bit semantics (§5.2) |
| `actions.width_policy` | `demos.load_demos` | `strict` refuses a corpus whose widths disagree with the profile; `coerce` truncates/pads with a printed warning |
| `input.deadzone`, `input.camera_half_deadzone`, `input.camera_full_deadzone`, `input.trigger_threshold` | `input_map` | analog thresholds deciding which axis and trigger bits fire |
| `input.keyboard_keys` | `input_map` | per-action-name keyboard stand-ins, used when no physical pad is connected |
| `deploy.fps` | `agent.cli.deploy` | enforced inference rate |
| `deploy.aggressiveness`, `deploy.attack_buttons` | `agent.cli.deploy`, `PolicyRunner.sharpened` | probability sharpening on the named bits — reachable now that the lookup reads the right mapping |
| `recording.max_trajectories` | `agent.cli.record` | stop after N saved files |
| `TRAINING_CONFIG.epochs` / `batch_size` / `learning_rate` | argparse defaults in `agent.cli.train` | overridden by `--epochs`, `--batch`, `--lr` |
| `TRAINING_CONFIG.window_size` | `architectures.policy_kwargs_for` → `TemporalAttentionLSTM` | recurrent feature-buffer length |
| `TRAINING_CONFIG.dagger_iterations` | `agent.cli.dagger --rounds` default | collect-then-retrain iterations |

**Deleted as meaningless:** `capture.width` / `capture.height` (the grab rectangle is derived from the
live window, so these never described anything the code did) and `TRAINING_CONFIG.demo_path` /
`model_path` / `train_path` (paths are derived from the profile by `agent/utils/paths.py`, so a relative
string could only ever be wrong).

---

## 6. Demonstration corpus

### 6.1 Recording protocol

`agent/cli/record.py` opens the environment against the live game, and on each loop iteration
reads (a) the captured frame and (b) the physical controller state, and appends the pair. Recording
starts and stops on the `K` key, and **the trajectory is written only when `K` stops it**: `ESC` merely
breaks the loop and the `finally` block only prints, so exiting with `ESC` while a recording is in
progress **discards the buffered frames**. (Earlier documentation stated that `ESC` saves; it does not —
see [`docs/TRAINING_GUIDE.md §3.1`](docs/TRAINING_GUIDE.md#31-protocol).) The buffer is serialised as a
Python list holding a single `imitation.data.types.Trajectory` with fields `obs`
`(N+1, 4, 128, 128) uint8`, `acts` `(N, num_actions) float32`, `infos = None` and `terminal = False` —
note the deliberate off-by-one: the last observation is the outcome of the last action, produced by
appending the first frame twice.

### 6.2 Measured corpus statistics

Counted directly from the `.pt` files present on the working machine (these files are **not** version
controlled — §15.2):

| Package | File | Transitions (`acts`) | Action width | Notes |
|---|---|---|---|---|
| `generic_agent` | `demo_0_20260523_221206.pt` | 6,301 | 18 | |
| `generic_agent` | `demo_1_20260523_221830.pt` | 7,068 | 18 | |
| `generic_agent` | `demo_0_20260523_224241.pt` | 8,149 | **7** | recorded after a reconfiguration |
| `generic_agent` | `demo_1_20260523_224751.pt` | 5,643 | **7** | recorded after a reconfiguration |
| `generic_agent` | **total** | **27,161** | mixed | ≈ 7.5 min at 60 fps |
| `hajime_agent` | `demo_0_20260619_103226.pt` | 6,150 | 18 | a single session |

All five files now live together in `runs/hajime_ippo/demos/`; the first column records which of the two
former packages each was recorded under, which is still the relevant provenance (§6.3). Each stored
`obs` array has one row more than its `acts` array (`N+1` frames for `N` transitions), so observation
counts are 6,302 / 7,069 / 8,150 / 5,644 / 6,151. Full per-file detail, including distinct joint-action
counts and entropies, is in [DATA.md §3](docs/DATA.md#3-measured-corpus-inventory).

Per-action marginal firing rates, pooled over the four `generic_agent`-era files after reconciliation to 18 bits:

| Index | Semantic | Firing rate |
|---|---|---|
| 0 | up / `W` | 0.4953 |
| 1 | down / `S` | 0.3748 |
| 2 | left / `A` | 0.0906 |
| 3 | right / `D` | 0.2070 |
| 4 | A / Cross / `SPACE` | 0.1245 |
| 5 | B / Circle / `F` | 0.0062 |
| 6 | X / Square / `R` | 0.0038 |
| 7–17 | triggers, stick press, camera | **exactly 0.0 in every file** |

Three structural properties follow, and all three bound the interpretation of §10:

1. **The corpus is small.** 27,161 transitions is the entire supervised signal, for a 128×128 pixel
   input with 18 output bits.
2. **The effective action space is far smaller than declared — and this is a code limitation, not a
   recording oversight.** Bits 7–17 are never actuated because the recorder only ever *writes* to indices
   0–6: in gamepad mode it maps the arrow keys and `i`/`o`/`p`, plus the D-pad mask and left stick, to
   bits 0–6, and an inline comment marked the rest as unmapped. Triggers, stick press and the whole
   discretised camera axis were therefore **unrecordable**, which is why no existing demonstration
   exercises them. `agent/utils/input_map.py` now maps all 18 bits onto physical controls, so they are
   recordable — but the corpus in §6.2 predates that fix. Only 45
   distinct joint action vectors occur in the whole generic pool and 24 in the Hajime session, out of
   2¹⁸ possible. The models are asked to predict 11 provably always-zero bits, which lowers achievable
   loss for architecture-independent reasons and compresses every cross-architecture comparison toward
   the floor.
3. **Class imbalance is severe, and a trivial predictor beats every model so far.** Two directional bits
   account for ~87 % of all actuations while two others fire below 1 %. Consequently a predictor that
   ignores the screen and reproduces only these marginals attains 2.6066 nats — better than every
   architecture in §10.1 (§10.3).

### 6.3 A data-integrity issue, disclosed and now blocked

`runs/hajime_ippo/demos/` mixes trajectories of width 18 with trajectories of width 7 — two files each —
because the corpus was recorded across a reconfiguration. The published benchmark in §10 reconciled both
to `num_actions = 9` by **silently truncating the 18-wide vectors and zero-padding the 7-wide ones**.
Those operations preserve the positional semantics of indices 0–6, so the resulting labels are internally
consistent, but the reconciliation was unlogged and sat inside a `try/except` that also dropped unreadable
files with a one-line message.

Consequences worth stating plainly:

1. **The benchmark's action space is neither profile.** It was a 9-bit space assembled under the old
   `generic_agent` package, whose own configuration targeted Roblox with keyboard/mouse. The
   "generic" label described the code path, not the data: the demonstrations are Hajime recordings.
2. **A strict reproduction now refuses this corpus.** Loading defaults to `width_policy = "strict"`, which
   raises naming the file and both widths. To re-obtain the published numbers you must state the
   reconciliation explicitly (`--width-policy coerce` with a 9-action profile); to do fresh work you should
   re-record. Corrupt files are likewise now a hard error listing every failure rather than a skip.
3. **Bits 7–17 are unreachable in the current corpus partly by code, not only by operator choice:** the
   old gamepad recorder wrote only indices 0–6. `agent/utils/input_map.py` now maps triggers, stick press
   and both camera axes onto their bits, so they are recordable — but no existing demonstration exercises
   them, and none of the §10 results involve them.

### 6.4 Reproducing the corpus statistics in §6.2

```bash
python - <<'PY'
from agent.config import load_profile, num_actions
from agent.utils import demos, paths
profile = load_profile("hajime_ippo")
cfg = profile["GAME_CONFIG"]
trajs = demos.load_demos(paths.demos_dir("hajime_ippo"), num_actions(cfg), policy="coerce")
demos.print_summary(demos.summarise(trajs, num_actions(cfg)))
PY
```

`summarise()` returns the frame count, per-bit marginals and entropies, the distinct-joint-action count,
the modal and top-5 shares, and both reference losses. `agent.cli.train` prints the same block before
training and logs it into MLflow, so a result can no longer be separated from the corpus that produced it.

---

## 7. Policy architectures

All six encoders implement Stable-Baselines3's `BaseFeaturesExtractor` contract: consume
`(B, 4, 128, 128)` and emit a `features_dim`-length vector that a shared `ActorCriticCnnPolicy` head
turns into multi-binary logits. Input pixels are cast to float and, where the encoder does it
explicitly, scaled to `[0,1]` then to `[-1,1]`.

### 7.1 NatureCNN (reference)

Stable-Baselines3's default encoder, selected by leaving `policy_kwargs` empty: three convolutions
(8/16/32 channels, kernels 8/4/3, strides 4/2/1) with ELU, then flatten to 512. Not re-implemented
here; it is the cheap baseline against which the custom encoders are judged.

- **Parameters:** 4,196,810 total policy ** 16.33 MB
- **Strength:** an order of magnitude cheaper to train than anything else in the study (§10).
- **Weakness:** no explicit temporal model beyond the 4 stacked frames; each decision is
  memoryless.

### 7.2 CNN + LSTM + temporal attention (`TemporalAttentionLSTM`, `utils/game_env.py:379-523`)

```
(B,4,128,128)
  -> 5 conv blocks: 32/64/128/256/512 channels, BatchNorm + ReLU, strides 2,2,2,1,2
  -> AdaptiveAvgPool2d(1,1) -> 512-d frame feature
  -> deque(maxlen=10) feature buffer  (10 timesteps)
  -> bidirectional LSTM, 2 layers, hidden 256, dropout 0.2  -> 512-d per step
  -> additive attention: Linear(512->512) -> Tanh -> Linear(512->1) -> softmax over time -> weighted sum
  -> Linear(512->1024) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(1024->512) -> ReLU
```

- **Parameters:** 6,116,779 ** 23.71 MB.
- **Design intent:** boxing sequences are non-Markovian — a punch is only meaningful relative to the
  opponent's animation a few frames earlier.
- **Caveats:** (i) the deque-based buffer makes the module **stateful across calls**, so its output
  depends on call order and batch size; `reset_hidden()` exists but is not invoked by the BC loop,
  and no episode boundary is ever defined (§5.3). (ii) `sequence.requires_grad_(True)` in-place
  re-enables gradients on detached tensors. (iii) A `SpatialAttention` module is defined immediately
  above it and is referenced by nothing.

### 7.3 Vision Transformer (`utils/vision_transformer.py`)

```
(B,4,128,128) -> PatchEmbedding: Conv2d(4 -> 256, k=s=16) -> 64 patches/frame
              -> + learnable CLS token + spatial embedding + temporal embedding (per source frame)
              -> 4 x TransformerEncoderBlock (pre-norm MHA, 4 heads, GELU FFN, mlp_ratio 2.0, drop 0.1)
              -> LayerNorm over CLS -> Linear(256 -> features_dim)
```

Sequence length 257 (1 CLS + 256 patches). Self-attention is applied jointly over space and time in
a single token set, which is precisely the inductive bias a CNN lacks: the relative phase of two
sprites far apart on screen is reachable in one layer.

- **Parameters:** 2,448,010 ** 9.70 MB — the second-smallest model, and by far the most
  parameter-efficient transformer.
- **Caveat:** quadratic cost over 257 tokens makes it the slowest encoder in the study by a wide
  margin (§10).

### 7.4 Impoola-CNN — residual encoder with global average pooling (`utils/impoola_cnn.py`)

The project's own contribution, motivated by the parameter cost of the flatten layer in §7.5.

```
Conv2d(4 -> 32, 3x3) -> ReLU -> MaxPool2d(2)
Conv2d(32 -> 64, 3x3) -> ReLU -> MaxPool2d(2)
Conv2d(64 -> 128, 3x3) -> ReLU -> MaxPool2d(2)
   each stage followed by 2 residual blocks (conv 3x3 + skip)
-> AdaptiveAvgPool2d(1,1)                 # <-- GAP replaces the flatten
-> Linear(128 -> features_dim=512)
```

- **Parameters:** 1,009,258 ** 4.20 MB — 5.8 % of the original Impala-CNN's size for a *lower*
  recorded loss (§10).
- **Why it works here:** the projection cost of a flatten head scales with spatial area
  (`16 × 16 × 128 = 32,768` inputs), whereas a GAP head costs `128`. At 128×128 input the flatten
  head is the dominant parameter block of the encoder, and it is exactly the block that generalises
  worst on a ~27 k-frame corpus.
- **What is traded away:** GAP destroys the spatial layout of the final feature map. Tasks that need
  absolute screen coordinates (reading a health bar's end point, tracking a millimetre-wide HUD
  element) lose information that the flatten variant retains. Nothing in this project measures that
  cost, because no such task exists in the evaluation set (§12).

### 7.5 Impala-CNN (original, flatten) (`utils/new_architectures.py:50`)

The Espeholt et al. (2018) "simple" convolutional network: the same three conv/pool/residual stages
as §7.4, but the final map is flattened to 32,768 and projected by a single dense layer to
`features_dim`. 17,720,938 parameters ** 67.95 MB — the heaviest model in the study.

### 7.6 ResNet-18 (`utils/new_architectures.py:98`)

`torchvision.models.resnet18(weights=None)`, with `conv1` replaced by a same-stride
`Conv2d(4, 64, 7, 2, 3)` to accept stacked frames, `fc` replaced by `Identity`, and a
`Linear(512 -> features_dim)` head. Weights are **not** pretrained: on a grayscale 128×128 input the
ImageNet statistics would not transfer, so this measures the *architecture* rather than
transfer learning. 11,516,938 parameters ** 44.37 MB.

ResNet-18 is the depth at which this family was stopped deliberately. ResNet-50 and above multiply
memory and inference latency for a 128×128 input whose spatial information content is already
saturated, and would overfit a 27 k-frame corpus (§12.2).

### 7.7 Attempted and withdrawn architectures

Three further encoders appear in the tracking store but are not in the reported comparison.

| Encoder | Runs recorded | Source status | Outcome |
|---|---|---|---|
| `Swin_Transformer` | 1 (2 epochs) | deleted | `RUNNING` forever in metadata; one logged batch, `bc/loss` 6.236 |
| `ConvNeXt` | 2 (10 epochs) | deleted | one failed after 93.7 s with no metrics; one `RUNNING` at `bc/loss` 6.235 |
| `Vision_Mamba` | 6 (1 epoch, batch 64) | deleted; only `__pycache__/mamba_architectures.cpython-311.pyc` survives | four `RUNNING` at `bc/loss` 6.226–6.241, two empty |

All nine of those runs sit at `bc/loss` ≈ 6.23 with `prob_true_act` ≈ 0.00195 ≈ 2⁻⁹ — the uniform
baseline for a 9-bit `MultiBinary` space. In other words **they never got past initialisation**, so the
store contains no evidence at all about these three architectures, positive or negative; see
[`docs/RESULTS.md §7`](docs/RESULTS.md#7-abandoned-architectures).

The documented reason for removing Swin and ConvNeXt is cost: at ~28 M parameters each with
batch 384 over stacked frames, they thrash local VRAM and extrapolate to roughly 40 h for a 10-epoch
run on the development GPU, which precludes iterating locally. That justification is credible but
**unquantified** — no memory or throughput measurement was retained, and no run in the tracking store
completed. `Vision_Mamba` is unreproducible from the current tree: its source module is gone and only
its compiled bytecode remains. §12.5 and [`docs/RESULTS.md`](docs/RESULTS.md) carry these as open
items rather than as findings.

---

## 8. Training objectives and algorithms

### 8.1 Behavioural cloning (primary)

All training goes through one entry point, `agent.cli.train`, which instantiates
`imitation.algorithms.bc.BC` over a Stable-Baselines3 `ActorCriticCnnPolicy` backbone and uses the
built-in sigmoid-bernoulli objective: for each of the `num_actions` independent bits, minimise
`-log π(a_human | s)`. The reported `bc/loss` is therefore the **sum over bits** of negative
log-likelihood, not a mean — which is why the cross-entropy floor for an 18-bit space with 11 dead bits is
not zero, why absolute loss values are not comparable across profiles with different `num_actions`, and
why §10.3 states the reference points explicitly.

MLflow captures, per logged batch: `bc/loss`, `bc/neglogp`, `bc/prob_true_act` (joint probability
assigned to the human's exact action vector), `bc/entropy`, `bc/ent_loss`, `bc/l2_norm`, `bc/epoch`,
`bc/batch`, `bc/samples_so_far`. `bc/l2_loss` is no longer written to the store: with
`l2_weight = 0.0` (the `imitation` default) it is identically zero, so `agent/utils/tracking.py` excludes
it and keeps the informative `bc/l2_norm`. Every run also logs dataset provenance — frame count,
trajectory count, action width, distinct joint-action count, and both reference losses — so a number
cannot be separated from the corpus that produced it. **No accuracy metric is computed anywhere**; earlier
documentation claimed MLflow logged one, and it does not (§12.1).

Warm start: `python -m agent.cli.train --arch <name> --model_path runs/hajime_ippo/models/<file>.zip`
loads the checkpoint and continues optimisation from it (covered by
`tests/test_train_smoke.py::test_warm_start_loads_the_previous_checkpoint`).

### 8.2 DAgger (collection complete, aggregation automated)

DAgger (Ross et al., 2011) addresses BC's compounding-error problem by querying the expert on states the
*learner* visits. `agent.cli.dagger` runs the whole loop: the policy plays, the human holds `L` to take
over, the steered segments are written as ordinary demonstrations into `runs/<profile>/demos/`, and the
corpus is then re-trained over — automatically, for `--rounds` iterations (defaulting to
`TRAINING_CONFIG.dagger_iterations`). Because corrections land in the same directory the trainer reads,
aggregation is "retrain over `demos/`" by construction rather than a separate mechanism.

The previous scripts advertised `train_agent.py --dagger`, whose handler was a `pass` statement with a
`TODO`, and no DAgger run has ever been recorded in the tracking store. That flag is gone; the loop is
real, and it still requires a running game, so it remains **empirically unvalidated** (§12.5).

### 8.3 GAIL (implemented, never completed)

`agent.cli.train_gail` wires `imitation.GAIL` with a `BasicRewardNet` discriminator and a PPO generator
(`batch_size 64`, `lr 3e-4`, `n_steps 1024`, `ent_coef 0.01`, `gamma 0.99`; `demo_batch_size 64`,
generator buffer capacity 2048, 4 discriminator updates per round — all now exposed as flags). Because the
generator must interact with the real window, the script runs the environment non-dummy and therefore
cannot run headless or on Linux. No GAIL run appears in the tracking store, and the adversarial path has
never been demonstrated end to end. It is also the only pipeline with no automated test, since nothing
short of a live game exercises it.

---

## 9. Experimental protocol

`agent.cli.benchmark` is the only script that produces the headline comparison. Its protocol:

| Item | Value |
|---|---|
| Demonstrations | all `demos/demo*.pt` in the working directory, concatenated |
| Transitions | 27,161 across 4 trajectories (§6.2) |
| Objective | sigmoid-bernoulli BC negative log-likelihood |
| Epochs | 10 (default, CLI-overridable) |
| Batch size | 384 |
| Learning rate | 1e-4, constant (`lr_schedule=lambda _: lr`) |
| Optimiser | Adam as configured by `imitation.BC` |
| RNG | `np.random.default_rng(seed=42)` (`agent/cli/train.py:SEED`) |
| Device | `cuda` |
| Per-model budget | one run per architecture (no repetitions in the headline table) |
| Tracked | `final_loss`, `training_time_s`, `model_size_mb`, `num_params` |
| Artefacts | `models/<Name>_policy.zip`, MLflow experiment `Model_Comparison`, regenerated tables |

Invocation:

```bash
python -m agent.cli.benchmark --profile hajime_ippo --epochs 10 --batch 384 --lr 1e-4
python -m agent.cli.benchmark --archs impala resnet18   # retrain a subset; the rest is read back from the store
```

When an architecture is not retrained, the benchmark reads its column back from the MLflow store
(`historical_from_store`) and reports how many completed runs that column summarises. If nothing is
found, the column renders as `-` with `Runs available = 0`. The previous harness instead substituted
hard-coded constants, and those had drifted until they disagreed with the report they were generating
— so a partial run silently published numbers from no known measurement. Nothing is now fabricated,
which means a partial sweep is safe to publish.

---

## 10. Results

### 10.1 Headline benchmark

Single-run values as recorded by the old `compare_models.py` in MLflow experiment `Model_Comparison`
and regenerated into `runs/hajime_ippo/models/comparison_results.md`. **All losses are in-sample
training NLL (§8.1); they are not generalisation estimates.**

<!-- BENCHMARK_START -->
| Metric | NatureCNN (reference) | CNN + LSTM + Attention | Vision Transformer (ViT) | Impoola-CNN (GAP) | Impala-CNN (Original) | ResNet-18 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Learning power (final loss, ↓)** | `3.48` *(Worst)* | `3.02` | `3.00` | `2.93` | `3.03` | **`2.90`** *(Best)* |
| **Training time (↓)** | **`18.3 seconds`** *(Fastest)* | `1.1 minutes` | `54.8 minutes` *(Slowest)* | `3.8 minutes` | `5.8 minutes` | `4.2 minutes` |
| **Artifact size (↓)** | `16.33 MB` | `23.71 MB` | `9.70 MB` | **`4.20 MB`** *(Smallest)* | `67.95 MB` *(Largest)* | `44.37 MB` |
| **Parameters (↓)** | `4.20 million` | `6.12 million` | `2.45 million` | **`1.01 million`** *(Most efficient)* | `17.72 million` *(Least efficient)* | `11.52 million` |
<!-- BENCHMARK_END -->

Higher-precision values and the corresponding joint action probabilities:

| Encoder | Final loss | Train time | Parameters | Artifact | `prob_true_act` at final logged batch |
|---|---|---|---|---|---|
| NatureCNN | 3.4763 | 18.3 s | 4,196,810 | 16.33 MB | 0.037 |
| CNN+LSTM+Attn | 3.0245 | 67.8 s | 6,116,779 | 23.71 MB | 0.068 |
| ViT | 2.9983 | 3,288.8 s | 2,448,010 | 9.70 MB | 0.067 |
| **Impoola-CNN (GAP)** | **2.9331** | 226.2 s | **1,009,258** | **4.20 MB** | 0.070 |
| Impala-CNN (Flatten) | 3.0276 | 345.1 s | 17,720,938 | 67.95 MB | 0.067 – 0.070 |
| **ResNet-18** | **2.9017** | 251.3 s | 11,516,938 | 44.37 MB | 0.068 – 0.075 |

### 10.2 Repeated-run dispersion

The tracking store contains more runs than the headline table admits. Repeats of the same
architecture at the same budget:

| Encoder | Runs with completed metrics | Final-loss spread | Time spread |
|---|---|---|---|
| NatureCNN | 2 | 3.4763 – 3.5817 (0.105) | 18.3 – 20.5 s |
| CNN+LSTM+Attn | 1 | — | — |
| ViT | 1 | — | — |
| Impoola-CNN | 1 | — | — |
| Impala-CNN | 4 | 2.9454 – 3.0298 (0.084) | 345 – 536 s |
| ResNet-18 | 3 | 2.9017 – 3.0189 (0.117) | 251 – 442 s |

**This table changes the reading of §10.1.** The gaps used to rank the four best encoders
(2.90 – 3.03) are of the same magnitude as the run-to-run spread of a *single* encoder (up to 0.12).
With `n = 1–4`, unpaired, and no variance reported, the ordering in §10.1 is **not** a statistically
supported ranking. The only claims that survive dispersion are (i) NatureCNN is clearly worse than
every custom encoder, and (ii) parameter count and training time differ by more than an order of
magnitude, which no seed noise can explain.

### 10.3 Reference baselines — the result that reframes the table

Because the objective is a summed per-bit negative log-likelihood, the loss of the best
*frame-independent* predictor is computable in closed form: a product-of-sigmoids that merely reproduces
each action bit's empirical marginal scores `Σᵢ H(mᵢ)` nats. Computed from the 9-bit reconciled
demonstration labels used by this benchmark (27,161 frames, [§6.2](#62-measured-corpus-statistics)):

| Predictor | In-sample NLL (nats) |
|---|---|
| Uniform (all logits 0) | `9 · ln 2` = **6.2383** |
| Per-bit action marginal (no conditioning on the frame at all) | **2.6066** |
| Best benchmark model (ResNet-18) | 2.9017 |
| Worst benchmark model (NatureCNN) | 3.4763 |

**Every architecture in §10.1 is worse on its own training data than a predictor that ignores the screen
entirely and only reproduces the action frequencies.** The gap is +0.30 nats for the best model and +0.87
for the worst. The uniform figure also confirms the accounting: the abandoned Swin/ConvNeXt/Vision-Mamba
runs logged their first `bc/loss` between 6.226 and 6.241 — they stopped at untrained initialisation.

This is a *budget* artefact rather than an architectural verdict. A 300-epoch run on the same corpus
reaches 1.330 nats, i.e. 1.28 nats **below** the marginal bound, which is where genuine visual-motor
conditioning has demonstrably begun. The benchmark simply terminates before the regime in which
architectures differ.

### 10.4 Longer training, for reference

`runs/hajime_ippo/logs/progress.csv` retains one historical run trained far
past the benchmark budget: 293 epochs (`bc/epoch` 0 → 292), 7,872,384 samples, `bc/loss` 4.851 → 1.330
and `bc/prob_true_act` 0.0078 → 0.483. Read alongside §10.3, it shows the loss values in §10.1 are very
early on the optimisation curve — a 10-epoch budget is roughly 3 % of the training this corpus can
absorb.

---

## 11. Analysis

**Parameter efficiency.** Impoola-CNN is the study's clearest engineering result. Replacing the flatten
projection with global average pooling removes 94 % of the checkpoint (67.95 MB → 4.20 MB) and 94 % of
the parameters (17.72 M → 1.01 M) while *improving* recorded loss (3.028 → 2.933) and cutting training
time by a third. On a corpus of this size that is the expected outcome: the flatten head is the block
most able to memorise 27 k frames, and GAP acts as a structural regulariser. Two qualifications are
required, though: the loss improvement lies entirely in the region *above* the marginal predictor's
2.6066 nats (§10.3), so it evidences faster approach to the prior rather than better conditioning; and
the residual block ordering, not the pooling choice alone, is a confound to untangle before attributing
the whole gain to GAP (§12.2).

**Fit quality.** Every encoder plateaus at `prob_true_act` ≈ 0.03–0.08, i.e. the joint action vector
the human produced is assigned under 8 % probability. More decisively, all six sit *above* the 2.6066
nats that a frame-independent per-bit marginal predictor attains on the same labels (§10.3), so at this
budget no encoder has been shown to condition on the screen at all. The corpus explains why the ceiling
is low: only 45 distinct joint actions occur in 27,161 frames, and 11 of the 18 action bits never fire,
so a large part of the achievable loss reduction is simply learning the prior. §10.4 shows the budget,
not the architecture, is the proximate cause.

**Cost.** ViT is the outlier: 2.45 M parameters — the second smallest model — yet 55 minutes of
training, 29× the Impala-CNN's wall time, because attention over 257 tokens is compute-bound rather
than parameter-bound. For real-time inference the checkpoint size that matters is the one §10.1
reports; for the *research loop* the binding cost is training time, and ViT is effectively out of
reach for iteration on this hardware.

**Depth of the temporal model.** The two encoders designed to capture temporal structure
(CNN+LSTM+Attention, ViT) do not beat the purely spatial Impoola-CNN. The most plausible explanation
is that stacked frames already carry short-range motion information and that 10 epochs is too few for
recurrent or attentional mechanisms to pay off — consistent with §10.3, where the loss is still falling
steeply at epoch 10.

**Conclusion**: the lowest in-sample training loss was reached by ResNet-18 (2.90 loss, 44.37 MB, 11.52M params), and the most size- and parameter-efficient model was Impoola-CNN (GAP) (2.93 loss, 4.20 MB, 1.01M params). These losses are training objectives measured on the demonstration corpus itself, so they rank optimisation ease rather than policy quality; see the threats-to-validity section before treating this as a ranking.

Impoola-CNN is the recommended default *when checkpoint size or inference memory is the constraint*;
ResNet-18 is the recommended default *when the lowest achievable loss is the constraint and 44 MB is
affordable*. Both statements are conditional on the 10-epoch budget and the absence of held-out
evaluation.

---

## 12. Threats to validity

Read this section before citing any number in this repository. Items are marked **open** (a limitation
of the evidence, unchanged), **mitigated** (the tooling now prevents it, the existing numbers are still
affected), or **fixed** (resolved in this revision, with the test that pins it). The full audit is in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

### 12.1 Open — limits on what the results mean

1. **No held-out evaluation, in any form.** There is still no validation split, no per-action accuracy,
   no closed-loop rollout, no success rate and no comparison against the human demonstrator. Every
   reported loss remains a training objective on the fitting data, and the architectures are still
   compared on *optimisation ease toward the action prior*, not on policy quality. Adding a split is a
   data-and-code task, not a documentation task; see §12.4.
2. **No statistical support.** 1–4 unpaired runs per architecture, one NumPy seed, no `torch.manual_seed`
   or `cudnn.deterministic`, and inter-architecture gaps (§10.2) no larger than intra-architecture
   spread. Any ranking of the top four encoders is provisional.
3. **The published numbers sit above the trivial baseline.** Once the marginal reference is computed
   (§10.3) every published figure turns out to be *worse than a predictor that ignores the screen*. The
   baseline is now logged automatically for every new run, but the historical numbers in §10.1 are
   unchanged and still carry that defect.
4. **The corpus is small, narrow and mislabelled.** 27,161 transitions; 11 of 18 action bits never fire;
   45 distinct joint actions; the Hajime session is 51 % no-op frames; and the widths are mixed (§6.3).
   None of that was re-recorded here, so it remains a limit on every existing result.
5. **Episodic metrics are structurally impossible.** The environment returns constant zero reward and
   never terminates (§5.3), so no success-rate number can be produced without first building a task
   signal. This is a design gap, not an omission.
6. **Retracted claims are still unmeasured.** The cost argument for withdrawing Swin and ConvNeXt has no
   retained memory or throughput measurement behind it, and `Vision_Mamba`'s source is gone; only its
   compiled bytecode survives (§7.7).
7. **Unexecuted pipelines.** GAIL has never completed a run and has no test (it needs a live game);
   DAgger's loop is now implemented but has still never been run; deployment logs nothing, so playing
   ability remains unevidenced.
8. **Data provenance.** Demonstrations, checkpoints and the MLflow store are git-ignored (§15.2), so the
   evidence behind §10 exists only on one workstation and the study is not externally reproducible.
9. **The recurrent extractor is stateful.** `TemporalAttentionLSTM` keeps a cross-call buffer and hidden
   state, so its output depends on call order and batch composition. Its numerics were deliberately left
   untouched so the CNN+LSTM row of §10.1 stays comparable; the property is documented in
   `agent/utils/temporal_lstm.py` and is **L12** in `docs/LIMITATIONS.md`.

### 12.2 Mitigated — the tooling now blocks the failure, the historical numbers still carry it

| Threat | Was | Now |
|---|---|---|
| Silent label corruption | action widths silently truncated/padded to the configured value | `width_policy = "strict"` raises, naming the file and both widths; `coerce` warns loudly. `tests/test_demos.py`, `tests/test_train_smoke.py` |
| Silent data loss | unreadable demo files skipped inside a bare `except` | a hard error listing every failure (`tests/test_demos.py::test_corrupt_file_raises_instead_of_shrinking_the_corpus`) |
| Silent capture failure | a dropped grab reused the previous frame with no trace | counted in `GenericGameEnv.dropped_frames`, and a window that never appears raises `WindowNotFoundError` instead of leaving a broken environment alive |
| No dataset provenance | runs did not record what they trained on | every run logs frame count, trajectory count, action width, distinct-joint count and both reference losses (§8.1) |
| Fabricated benchmark columns | `--only-new` substituted hard-coded constants that had drifted | untrained columns are read back from the store and reported as `-` with `Runs available = 0` (§9) |
| Contradicting conclusion prose | hand-written sentence claimed the wrong winner | generated from the measured values, with a regression test (`tests/test_benchmark_format.py`) |
| Unrecordable action bits | the gamepad recorder wrote only indices 0–6 | all 18 bits have a physical source in `agent/utils/input_map.py`; existing demonstrations still do not exercise 7–17 |

### 12.3 Fixed in this revision

- Two near-duplicate packages collapsed into `agent/` with per-title profiles (§3.1).
- Every configuration field is now consumed or deleted; eleven were previously decorative (§5.5).
- `actions.mappings` drives both emission and capture, replacing hard-coded index → button dispatch (§5.2).
- Frame pacing (`target_fps`, `deploy.fps`) and the `aggressiveness` sharpening path actually work (§5.5).
- `ESC` in the recorder flushes the in-progress trajectory instead of discarding it (§6.1;
  `tests/test_recorder.py`).
- The dead `./models/steps` checkpoint path, the `demos*.pt` glob bug and the misspelled
  `train_imiation.py` are gone; `get_last_index` sorts numerically and is tested (§3).
- Windows-only imports are behind feature flags, so `agent.utils.game_env` imports and dummy-mode
  training runs on Linux and without the ViGEmBus driver (§13.4).
- Dead code and metrics removed: `SpatialAttention`, `bc/l2_loss` in the store.
- CPU fallback is reachable (`--device` defaults to unset and is resolved at runtime).
- Machine-specific emulator/ROM paths moved out of version control into `agent/config/local.py`.
- A pytest suite (69 tests) and Linux CI exist, and the `Dockerfile` installs an interpreter. A
  linter/formatter configuration still does not, and the Docker image has still never been built.

### 12.4 The shortest path to a citable result

1. Split the corpus leave-one-file-out; report held-out per-bit accuracy, F1 and joint exact-match.
2. Report margin over `marginal_baseline_nats`, which the tooling now logs automatically.
3. Seed torch, run ≥5 seeds per encoder, publish mean ± SD.
4. Re-record with the now-recordable action bits, or reduce the declared space to what is used.
5. Train to the budget where models beat the baseline (the 300-epoch reference run does, at 1.330 nats).

Only then does §10.1 become a comparison of policies rather than of convergence speed.

---

## 13. Getting started

### 13.1 Requirements

- **Python 3.11.**
- **Offline work (training, benchmarking, tests) is portable**: it runs on Linux and on Windows without
  any capture or actuation dependency, because `agent/utils/windows.py` puts the platform imports behind
  feature flags and `dummy=True` never touches them. This is what CI exercises.
- **Live work (recording, deployment, GAIL) needs Windows 10/11** with `dxcam` (Desktop Duplication),
  `pywin32` (window enumeration) and, for gamepad profiles, `vgamepad` plus the **ViGEmBus** kernel driver.
- NVIDIA GPU with CUDA 12.x for any training run of practical length; `--device cpu` works for smoke
  tests and is what the tests and CI use.

### 13.2 Install

Fresh environment (recommended) — torch first, from PyTorch's CUDA index, because
`torch==2.5.1+cu121` carries a local version label that PyPI does not host:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-minimal.txt
pip install pytest            # to run the suite
```

`requirements-minimal.txt` is the curated list derived from the real import statements.
`requirements.txt` is the verbatim `pip freeze` of the workstation that produced the recorded results —
142 packages including MLflow's whole server stack, `tensorboard`, `optuna`, and `inputs==0.5` which
nothing imports. Reproduce the exact publication environment with `pip install -r requirements.txt
--extra-index-url https://download.pytorch.org/whl/cu121`; install the curated file for everyday work.

On Linux, use `requirements-docker.txt` and the CPU wheels:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-docker.txt
```

### 13.3 Verify the installation

```powershell
python -m pytest -q            # 69 tests, CPU only, no game required
python test_xinput.py          # enumerates XInput slots 0-3
```

`test_xinput.py` prints `res=0` for a slot with a connected **physical** controller and `res=1167`
(`ERROR_DEVICE_NOT_CONNECTED`) otherwise. All four slots returning 1167 means no pad is plugged in: the
recorder and DAgger then fall back to keyboard capture (`input.keyboard_keys`), which reaches every bit
except the analog stick, while a real controller is needed to exercise the left/right stick axes.

For gamepad actuation the **ViGEmBus** kernel driver must be installed separately — `vgamepad` is only a
binding for it. Check with `Get-Service ViGEmBus`; if it is missing, `import vgamepad` raises at import
time, and the environment module will not load outside dummy mode. On the machine this revision was
written on the service is running and `vgamepad.VX360Gamepad()` creates, updates and resets without
error, so recording and deployment are executable here; no live end-to-end session was run as part of
this work, so those paths remain exercised only by unit tests.

### 13.4 Docker

`Dockerfile` targets offline (dummy-mode) CUDA training. It installs Python 3.11 and the OpenCV runtime
libraries into the CUDA base image, resolves torch/torchvision from PyTorch's CUDA index and the rest from
`requirements-docker.txt`, and defaults to a one-epoch CPU training smoke run:

```bash
docker build -t imitation-player .
docker run --rm --gpus all -v "$PWD/runs:/app/runs" imitation-player \
  python -m agent.cli.train --profile hajime_ippo --arch impoola --epochs 10
```

**The image has never been built**, so this is a reasoned configuration rather than a verified one. What
is no longer true of it: the previous version could not start at all — a `-runtime` base with no
interpreter, and a dependency-stripping `sed` that left `game_env.py` importing `vgamepad` at module
scope, which made even dummy-mode training unimportable on Linux. Both are fixed: the imports are now
conditional, so the platform half of the problem is gone regardless of the image. Recording, deployment
and GAIL still cannot run in a container: they need a real window, an input device and a desktop.

---

## 14. Usage guide

The full operator-facing procedure — recording hygiene, what a healthy loss curve looks like, and how to
run DAgger corrections — is in [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md). Condensed, from the
repository root:

```bash
export IMITATION_PROFILE=hajime_ippo     # optional; --profile also works on every command

# 1. Launch the game/emulator first, then record.
#    K starts and stops a recording (stopping is what SAVES); ESC also flushes now.
python -m agent.cli.record

# 2. Behavioural cloning. Prints the corpus statistics and the trivial baselines before training.
python -m agent.cli.train --arch impoola --epochs 100 --batch 384 --lr 1e-4

# 3. Inspect the live loss curves.
mlflow ui --backend-store-uri file:runs/hajime_ippo/mlruns     # http://localhost:5000

# 4. Deploy. K toggles AI/human, ESC quits.
python -m agent.cli.deploy --arch impoola

# 5. DAgger: hold L to take over; corrections are written to demos/ and the model retrains, --rounds times.
python -m agent.cli.dagger --arch impoola --rounds 3

# 6. Re-run the architecture study (regenerates §10.1 and comparison_results.md).
python -m agent.cli.benchmark --epochs 10 --batch 384 --lr 1e-4

# 7. GAIL - experimental, needs the live game, never completed a run here.
python -m agent.cli.train_gail --timesteps 100000
```

Recording practice that the results in §10 depend on: **variety beats volume.** Short clips
(1–2 min) covering distinct situations — offence, defence, whiffed combos, being cornered — produce a
usable action distribution; long unbroken sessions do not. The corpus statistics in §6.2 show what
happens otherwise: 11 of 18 action bits never fired.

---

## 15. Reproducibility

### 15.1 Determinism

`agent/cli/train.py:SEED = 42` seeds the NumPy generator that shuffles demonstrations, and every run logs
that seed as an MLflow parameter. No `torch.manual_seed`, `cudnn.deterministic` or
`torch.use_deterministic_algorithms` call exists, so CUDA kernel nondeterminism remains uncontrolled —
which §10.2 shows is material at this sample size. The recorded results therefore cannot be reproduced
bit-for-bit, only distributionally. Nothing pins the Python or CUDA build beyond `requirements.txt`.

### 15.2 What is and is not versioned

| Artefact | Path | In git? |
|---|---|---|
| Source code, tests, docs, CI | `agent/`, `tests/`, `docs/`, `.github/` | yes |
| Configuration profiles | `agent/config/profiles/*.py` | yes |
| Machine-specific paths | `agent/config/local.py` | **no** (git-ignored; `local.example.py` is the template) |
| Generated comparison report | `runs/hajime_ippo/models/comparison_results.md` | yes |
| One historical training log | `runs/hajime_ippo/logs/progress.csv` | yes |
| Human demonstrations | `runs/*/demos/*.pt` (~3.6 GB) | **no** (`*.pt`) |
| Trained checkpoints | `runs/*/models/*.zip` | **no** (`*.zip`; LFS-filtered if force-added) |
| Reference BC policy | GitHub release asset `bc_policy.zip` (17 MB) | **no** — no longer a git blob |
| MLflow store (all 32 recorded runs) | `runs/*/mlruns/` | **no** |

The repository previously carried `bc_policy.zip` as a 17 MB git blob with Git LFS unconfigured. It has
been removed from tracking and published as a release asset instead, with LFS configured for `*.zip` so
that any checkpoint deliberately added later goes to LFS rather than into the object store:

```bash
curl -L -o runs/hajime_ippo/models/bc_policy.zip \
  https://github.com/PedroM2626/Imitation-player/releases/latest/download/bc_policy.zip
```

The consequence that has *not* changed: the demonstrations and the MLflow store are still single-machine,
so §10 cannot be independently reproduced. Checksums and an export procedure are in
[docs/DATA.md §7](docs/DATA.md#7-what-the-corpus-needs).

---

## 16. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `WindowNotFoundError` after the timeout | Game not running, or `process_name` does not match | Start the title first; set `process_name` to the exact substring of the owning process; the exception now names the timeout it used |
| DXCam warning, falls back to `mss` | Dual-GPU laptop; duplication attaches to the adapter that is not presenting the window | Expected on Optimus laptops; capture is slower — accept lower FPS or move the window to the primary adapter |
| Captured region is offset / includes the title bar | Window decorations included in the grab rectangle | Tune `window_offset.{left,top,right,bottom}` |
| `demo_*.pt: actions have width 7 but the profile declares num_actions=18` | The corpus mixes widths recorded before a reconfiguration | Use the profile that matches the data, re-record, or pass `--width-policy coerce` to accept truncation/padding with a warning |
| `import vgamepad` raises a `VIGEM_ERRORS` name at start-up | ViGEmBus driver not installed (the pip package does not install it) | Install ViGEmBus, reboot, confirm with `Get-Service ViGEmBus`. Training in dummy mode is unaffected: the import is now conditional |
| Virtual pad exists but the game ignores it | The title was started before the pad was attached, or polls DirectInput only | Start the game *after* the environment creates the pad |
| Training loss plateaus around 3.0 and never improves | 10-epoch budget on a 27 k-frame corpus | Train far longer; §10.4 reaches 1.33 loss at ~290 epochs |
| Loss improves but stays above ~2.61 nats (§hajime_ippo 9-bit corpus) | That is the per-bit marginal baseline — the model is reproducing the action prior | Train longer or rebalance the corpus; the trainer prints this reference before every run |
| Policy only ever presses two directions | Action marginals dominate (§6.2) and the corpus lacks variety | Record the missing actions explicitly; consider label re-weighting |
| `CUDA out of memory` | Batch 384 over 4×128×128 float, largest on flatten heads | Lower `--batch`, or prefer the GAP-based encoder |
| `ModuleNotFoundError: dxcam` / `pywin32` on Linux | Expected: live capture and actuation are Windows-only | Offline training, benchmarking and the test suite run fine on Linux; only recording, deployment and GAIL need Windows |
| `Unknown profile 'x'` | Typo or unset `IMITATION_PROFILE` | Available profiles are listed in the error; add one under `agent/config/profiles/` |
| `exe_path` is `None` and the game never starts | Paths were moved out of version control | Copy `agent/config/local.example.py` to `agent/config/local.py` and set `LOCAL_OVERRIDES` |
| A recording is lost after pressing ESC | No longer possible: ESC, window-close and Ctrl+C all flush the buffer | If a segment looks empty, it had fewer than two frames and was intentionally not written |

---

## 17. Documentation index

| Document | Covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Environment internals, capture/actuation paths, every feature extractor with tensor shapes, module dependency graph |
| [`docs/DATA.md`](docs/DATA.md) | Recording protocol, demonstration file format, corpus statistics, action-space integrity |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Protocol per script, CLI reference, MLflow metric definitions, how to add an architecture |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Full run-by-run results from the tracking store, dispersion analysis, trivial baselines, withdrawn architectures |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Every limitation and defect, marked open / mitigated / fixed, with a prioritised work list |
| [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) | Operator-facing step-by-step guide (English successor to the removed `GUIA_TREINAMENTO.md`) |
| [`docs/SETUP.md`](docs/SETUP.md) | Environment installation, Windows-only dependency map, Docker status |
| `tests/` + `.github/workflows/ci.yml` | The executable specification: config consistency, demo integrity and baseline arithmetic, emission semantics, checkpoint resolution, environment shapes, recorder flushing, benchmark formatting, one-epoch CPU training for all six encoders |

---

## 18. License and attribution

This repository is distributed under the **MIT License** (see [`LICENSE`](LICENSE)), copyright 2026 Pedro
Morato Lahoz. It was previously unlicensed; the license was chosen explicitly in this revision rather
than left implicit.

Attribution: the environment design, the recorder/trainer loop and the demonstration format follow the
work published by [@paulo101977](https://github.com/paulo101977/notebooks-rl), from which this project
was derived, and `GenericGameEnv` is an adaptation of that author's `resident_requiem.py`. The
`Impoola-CNN` encoder, the ViT extractor, the benchmark harness and the MLflow instrumentation are
original to this repository. Third-party components carry their own terms: PyTorch (BSD-3),
stable-baselines3 (MIT), `imitation` (Apache-2.0), MLflow (Apache-2.0), dxcam (MIT), vgamepad (MIT).
