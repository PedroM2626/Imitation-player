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
| [3. Repository layout](#3-repository-layout) | Package structure and the `generic_agent` / `hajime_agent` fork |
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
- An **automated architecture benchmark** (`compare_models.py`) that trains each encoder under an
  identical budget and writes the comparison into this README and into
  `generic_agent/notebooks/models/comparison_results.md`.
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
| Record | `notebooks/record_trajectories.py` | live game + human input | `notebooks/demos/demo_*.pt` |
| Clone | `notebooks/train_agent*.py` | `demos/*.pt` | `models/bc_policy*.zip`, MLflow run |
| Benchmark | `notebooks/compare_models.py` | `demos/*.pt` | 6 policies + generated tables |
| Deploy | `notebooks/run_ai*.py` | `models/*.zip` | live play through virtual pad |
| Correct | `notebooks/run_dagger.py` | live game + policy + human | additional demos |
| Adversarial | `notebooks/train_gail.py` | live game + demos | adversarially trained policy |

---

## 3. Repository layout

```
imitation-player/
│
├── hajime_agent/                 # Title-specific configuration: Hajime no Ippo (PS3 / RPCS3)
│   ├── config/
│   │   └── game_config.py        # Process name, ROM path, capture box, 18-action table
│   ├── utils/
│   │   ├── game_env.py           # Gymnasium env + TemporalAttentionLSTM extractor
│   │   ├── utils.py              # get_last_index, LSTMWrapper (recurrent inference shim)
│   │   ├── impoola_cnn.py        # Impoolа-CNN (GAP) feature extractor
│   │   ├── new_architectures.py  # Impala-CNN (Flatten) and ResNet-18 extractors
│   │   └── vision_transformer.py # ViT patch/pos/temporal-embedding extractor
│   ├── notebooks/
│   │   ├── record_trajectories.py  # Demonstration recording
│   │   ├── train_agent.py          # BC with MLflow + transfer learning
│   │   ├── train_agent_lstm.py     # BC over CNN+LSTM+Attention
│   │   ├── train_agent_transformer.py # BC over ViT
│   │   ├── train_agent_impoola.py  # BC over Impoola-CNN (GAP)
│   │   ├── train_imiation.py       # Minimal BC (no MLflow) — see §12.4, filename is a typo
│   │   ├── train_gail.py           # GAIL (adversarial imitation)
│   │   ├── run_dagger.py           # Interactive DAgger collection
│   │   ├── run_ai.py               # Deploy: CNN+LSTM policy in the live game
│   │   ├── run_ai_lstm.py          # Deploy: recurrent policy
│   │   ├── run_ai_transformer.py   # Deploy: ViT policy
│   │   └── compare_models.py       # Automated 6-architecture benchmark
│   ├── demos/                      # Recorded trajectories (.pt, not versioned)
│   ├── models/                     # Trained policies (.zip, not versioned)
│   └── mlruns/                     # Local MLflow store (not versioned)
│
├── generic_agent/                # Same pipeline, configured for an arbitrary title
│   ├── config/game_config.py     # Example: Roblox, keyboard/mouse input, 9 actions
│   ├── utils/                    # Same modules; game_env.py additionally implements
│   │                             #   actions.input_mode == "keyboard_mouse"
│   └── notebooks/                # Same scripts; all benchmark evidence lives here
│       └── models/
│           ├── bc_policy.zip               # Reference BC policy (the only model versioned)
│           └── comparison_results.md       # Generated by compare_models.py
│
├── docs/                         # Deep-dive documentation (see §17)
├── test_xinput.py                # Standalone XInput controller probe
├── requirements.txt              # Exact frozen environment (pip freeze, Windows/CUDA)
├── Dockerfile                    # CUDA training image — currently non-functional, §13.4
└── .gitignore
```

### 3.1 The two-package fork

`hajime_agent/` and `generic_agent/` are **two configurations of one code base**, not two libraries.
Nine of the nineteen Python files are byte-identical (`utils/utils.py`, `utils/impoola_cnn.py`,
`utils/new_architectures.py`, `utils/vision_transformer.py`, `notebooks/run_ai_lstm.py`,
`run_ai_transformer.py`, `train_agent_impoola.py`, `train_gail.py`, `train_imiation.py`). The rest
diverge in three places, all of which are input/output concerns rather than modelling:

| Diverging file | Nature of the difference |
|---|---|
| `config/game_config.py` | `RobloxPlayerBeta` + keyboard/mouse + 9 actions vs `rpcs3` + gamepad + 18 actions, plus hard-coded emulator/ROM absolute paths in the Hajime copy |
| `utils/game_env.py` | Only `generic_agent` implements the `keyboard_mouse` emission branch; `hajime_agent` is gamepad-only |
| `notebooks/run_ai.py` | Inference rate (`MAX_FPS` 120 vs 30); `generic_agent` adds a logit-boost "aggressiveness" branch |
| `notebooks/record_trajectories.py` | Action-vector width and the keyboard/mouse mapping path |
| `notebooks/train_agent.py` | `generic_agent` pads/truncates action vectors to the configured width (§6.3) |
| `notebooks/compare_models.py` | Duplicated hard-coded baseline numbers, which have drifted apart between the two copies |

Consequences worth stating: fixes applied to one package do not automatically propagate to the
other, and **all recorded benchmark evidence currently lives under `generic_agent/`**, so the Hajime
copy cannot be validated against the tables in §10. Consolidation into a single package plus a
per-title configuration file is the obvious refactor (§12.6).

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

`GenericGameEnv` (`generic_agent/utils/game_env.py`) subclasses `gymnasium.Env`. It is best read as a
**partially specified MDP**: the state and action channels are fully implemented, while reward and
termination are deliberately degenerate because BC does not consume them.

### 5.1 Observation space

```python
Box(low=0, high=255, shape=(128, 128, 1), dtype=uint8)   # single HWC grayscale frame
```

The model, however, never sees that space directly. The training scripts wrap the environment in
`DummyVecEnv -> VecTransposeImage -> VecFrameStack(n_stack=4)` (`compare_models.py:140-147`), so the
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
MultiBinary(num_actions)     # num_actions = 18 (hajime_agent) or 9 (generic_agent)
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

The 9-dimensional keyboard/mouse layout used by the `generic_agent` example configuration maps each
index to a `pydirectinput` key or mouse button through the `actions.mappings` table.

> **Known mismatch.** In gamepad mode the `mappings` table is *never read*; `step()` dispatches on
> hard-coded indices. The `vg_code` names declared in `hajime_agent/config/game_config.py`
> (`DS4_BUTTON_CROSS`, …) are therefore inert — the environment presses `XUSB_GAMEPAD_A/B/X`. Also,
> `TRAINING_CONFIG.window_size`, `dagger_iterations`, `demo_path`, `model_path`, `train_path` and all
> four `INPUT_CONFIG` fields are unused by the training scripts, which hard-code `./demos/` and
> `./models/` (§12.6).

### 5.3 Reward, termination, and the deployment loop

`step()` returns `observation, 0.0, False, False, {}`. Reward is constant zero and neither
`terminated` nor `truncated` ever becomes `True`; `reset()` clears only the pressed-key set. This is
sound for BC, which ignores both channels, but it means the environment as written **cannot support
any episodic evaluation metric** — there is no notion of a round ending, a life being lost, or a
match being won. Any future success-rate work requires a reward/terminal signal, which in a
black-box setting must be inferred from pixels (a HUD reader) or from the game's own state.

Inference-rate control is likewise external: the environment computes `self.frame_time` from
`target_fps` but never uses it, and pacing is imposed by the `run_ai*.py` loops (`MAX_FPS` 120 in
`generic_agent`, 30 in `hajime_agent`). Because capture is *asynchronous* to the emulator's own
frame rate, the recorded `(frame, action)` alignment is only as tight as the operator's reaction
time, and the same action may be logged across several consecutive frames.

### 5.4 Dummy mode

Setting `config["dummy"] = True` returns from `__init__` before window lookup, gamepad creation and
camera acquisition. This allows offline training on recorded demonstrations without a running game
and is what every training script uses. It is **not** a functioning offline environment: `step()`
still dereferences `self.gamepad`, so any code path that steps a dummy environment raises. GAIL,
which must generate fresh trajectories, therefore forces `dummy = False` and cannot run headless.

---

## 6. Demonstration corpus

### 6.1 Recording protocol

`record_trajectories.py` opens the environment against the live game, and on each loop iteration
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

Each stored `obs` array has one row more than its `acts` array (`N+1` frames for `N` transitions), so
observation counts are 6,302 / 7,069 / 8,150 / 5,644 / 6,151. Full per-file detail, including distinct
joint-action counts and entropies, is in [DATA.md §3](docs/DATA.md#3-measured-corpus-inventory).

Per-action marginal firing rates, pooled over the `generic_agent` corpus after reconciliation to 18 bits:

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
   bits 0–6, and an inline comment marks the rest as unmapped (`record_trajectories.py:122-155`).
   Triggers, stick press and the whole discretised camera axis are therefore **unrecordable with the
   current code**, so no demonstration corpus produced by this repository can ever train them. Only 45
   distinct joint action vectors occur in the whole generic pool and 24 in the Hajime session, out of
   2¹⁸ possible. The models are asked to predict 11 provably always-zero bits, which lowers achievable
   loss for architecture-independent reasons and compresses every cross-architecture comparison toward
   the floor.
3. **Class imbalance is severe, and a trivial predictor beats every model so far.** Two directional bits
   account for ~87 % of all actuations while two others fire below 1 %. Consequently a predictor that
   ignores the screen and reproduces only these marginals attains 2.6066 nats — better than every
   architecture in §10.1 (§10.3).

### 6.3 A data-integrity issue that must be disclosed

`generic_agent/notebooks/demos/` mixes trajectories of width 18 with trajectories of width 7, while
the package's `config` declares `num_actions = 9`. The loaders reconcile this by **silently
truncating the 18-wide vectors to 9 and zero-padding the 7-wide vectors to 9**
(`generic_agent/notebooks/train_agent.py`, `generic_agent/notebooks/compare_models.py:123-127`). Both
operations preserve the *positional* semantics of indices 0–6, so the resulting 9-bit labels are
internally consistent, but the reconciliation is unlogged and happens inside a `try/except` that
drops a failing file with a one-line message rather than aborting.

The practical reading: **the benchmark in §10 was run on a 9-bit action space assembled from Hajime
recordings**, and the "generic" Roblox configuration in `config/game_config.py` was never trained on.
The `generic_agent` label therefore describes the *code path* (input emission), not the *data*.

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

All training scripts instantiate `imitation.algorithms.BC` with a Stable-Baselines3
`ActorCriticCnnPolicy` backbone, `MsePolicy`-style *categorical* losses disabled and the built-in
sigmoid-bernoulli objective used instead: for each of the `num_actions` independent bits, minimise
`-log π(a_human | s)`. The reported `bc/loss` is therefore the **sum over bits** of negative
log-likelihood, not a mean — which is why the cross-entropy floor for an 18-bit space with 11 dead
bits is not zero and why absolute loss values are not comparable across configurations with
different `num_actions`.

MLflow captures, per logged batch: `bc/loss`, `bc/neglogp`, `bc/prob_true_act` (joint probability
assigned to the human's exact action vector), `bc/entropy`, `bc/ent_loss`, `bc/l2_norm`,
`bc/l2_loss`, `bc/epoch`, `bc/batch`, `bc/samples_so_far`. Note that `bc/l2_loss` is identically 0 in
the retained logs (the gradient-norm penalty term is disabled), and that no *accuracy* metric is
logged despite earlier documentation claiming one.

Transfer/warm-start is supported: `train_agent.py --model_path <dir>` loads an existing zip and
continues optimisation from it.

### 8.2 DAgger (partially implemented)

DAgger (Ross et al., 2011) addresses BC's compounding-error problem by querying the expert on states
the *learner* visits. `run_dagger.py` implements the data-collection half faithfully: the policy acts,
the human presses `L` to take over, and the human's inputs are appended to a new demonstration file
under `demos/`. The **re-aggregation and re-training half is a stub**:
`train_agent.run_dagger_iteration()` and `train_imiation.dagger_iteration()` contain only `pass` and
a `TODO`. DAgger is therefore a manual two-tool workflow today (collect, then re-run training), not
an automated loop, and no DAgger iteration was ever executed — there is no run in the tracking store
attributable to it.

### 8.3 GAIL (implemented, never completed)

`train_gail.py` wires `imitation.GAIL` with a `BasicRewardNet` discriminator and a PPO generator
(`batch_size 64`, `lr 3e-4`, `n_steps 1024`, `ent_coef 0.01`, `gamma 0.99`; `demo_batch_size 64`,
generator buffer capacity 2048, 4 discriminator updates per round). Because the environment cannot
step in dummy mode (§5.4), GAIL requires the game running and the operator's machine capturing the
window; the script forces `dummy = False`. No GAIL run appears in the tracking store, and the
adversarial path has never been demonstrated end to end.

---

## 9. Experimental protocol

`compare_models.py` is the only script that produces the headline comparison. Its protocol:

| Item | Value |
|---|---|
| Demonstrations | all `demos/demo*.pt` in the working directory, concatenated |
| Transitions | 27,161 across 4 trajectories (§6.2) |
| Objective | sigmoid-bernoulli BC negative log-likelihood |
| Epochs | 10 (default, CLI-overridable) |
| Batch size | 384 |
| Learning rate | 1e-4, constant (`lr_schedule=lambda _: lr`) |
| Optimiser | Adam as configured by `imitation.BC` |
| RNG | `np.random.default_rng(seed=42)` |
| Device | `cuda` |
| Per-model budget | one run per architecture (no repetitions in the headline table) |
| Tracked | `final_loss`, `training_time_s`, `model_size_mb`, `num_params` |
| Artefacts | `models/<Name>_policy.zip`, MLflow experiment `Model_Comparison`, regenerated tables |

Invocation:

```bash
cd generic_agent/notebooks
python compare_models.py --epochs 10 --batch 384 --lr 1e-4
python compare_models.py --only-new          # retrain a subset, reuse stored baselines
```

`--only-new` deserves a warning: when an architecture is not retrained, the script substitutes
**hard-coded** baseline constants rather than reading the tracking store, and those constants have
drifted out of agreement with the numbers now printed in `comparison_results.md`. Treat the
generated table as authoritative only for a full run.

---

## 10. Results

### 10.1 Headline benchmark

Single-run values as recorded by `compare_models.py` in MLflow experiment `Model_Comparison` and
regenerated into `generic_agent/notebooks/models/comparison_results.md`. **All losses are in-sample
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

`generic_agent/notebooks/models/imitation/bc_logs/progress.csv` retains one historical run trained far
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

Read this section before citing any number in this repository.

1. **No held-out evaluation, in any form — and no reference baseline.** There is no validation split, no
   per-action accuracy, no closed-loop rollout, no success rate, no comparison against the human
   demonstrator, and no independent test trajectories. All reported losses are training objectives on
   the fitting data. Worse, no marginal baseline was ever computed, and once it is (§10.3) every
   published number turns out to be *worse than a predictor that ignores the screen*. The architectures
   are compared on *optimisation ease toward the action prior*, not on policy quality.
2. **No statistical support.** 1–4 unpaired runs per architecture, one RNG seed, variance never
   reported in the headline table, and inter-architecture gaps (§10.2) no larger than intra-architecture
   spread. Any ranking of the top four encoders is provisional.
3. **Silent degradation paths.** Capture failures re-emit the previous frame; failed demonstration
   files are skipped inside `except Exception` with a one-line message; the action-width reconciliation
   in §6.3 changes label semantics without logging. Each of these can alter a result without
   producing an error.
4. **Documentation–code drift (two instances corrected in this revision).** Earlier revisions of this
   README claimed MLflow logged accuracy (it does not) and asserted that Impoola-CNN achieved the
   lowest loss (ResNet-18 did: 2.90 vs 2.93). The generated conclusion sentence in
   `compare_models.py` was likewise hard-coded and repeated that error; it is now derived from the
   measured values, and the stored fallback baselines were realigned with the recorded MLflow runs
   (the Impala and ResNet-18 entries had drifted to values matching no completed run). The failure
   mode remains: prose that is not regenerated from data can silently contradict the table above it.
5. **Unverified components.** GAIL has never completed a run; DAgger's re-training loop is a `pass`
   stub; Swin/ConvNeXt/Vision-Mamba attempts all died at the first logged batch and two of the three
   sources have been deleted; the `Dockerfile` cannot start (no Python in the CUDA runtime base image,
   and `import vgamepad` at module top makes `game_env.py` unimportable on Linux even in dummy mode);
   the repository's own `venv/` points at an interpreter that does not exist on this machine; the
   ViGEmBus driver is not installed here, so no recording or deployment path has been exercised on this
   machine at all. Two scripts also advertise a CPU fallback that cannot fire:
   `device = args.device or check_cuda()` is unreachable because `--device` defaults to the non-empty
   string `"cuda"`, so a CPU-only machine must pass `--device cpu` explicitly.
6. **Structural debt.** Two near-duplicate packages (§3.1), whose `compare_models.py` copies had
   divergent hard-coded baselines — these are now identical, which removes the drift but also means
   the Hajime copy reports the `generic_agent` corpus's numbers when run with `--only-new`; the Hajime
   demonstrations have never been benchmarked. Configuration fields are read by nothing;
   `hajime_agent/config/game_config.py` hard-codes absolute emulator and ROM paths;
   `train_imiation.py` is misspelled in both packages and globs `demos*.pt` where every other script
   and the recorder use `demo*.pt` (so it loads zero demonstrations); there is no test suite, no
   linter configuration and no CI.
7. **Data provenance.** Demonstrations, checkpoints and the MLflow store are all git-ignored
   (§15.2), so the evidence behind §10 exists only on one workstation. A reader cannot currently
   reproduce the study.

---

## 13. Getting started

### 13.1 Requirements

- **Operating system: Windows 10/11.** The capture and actuation stack is Windows-specific
  (`dxcam` uses Desktop Duplication; `pywin32` enumerates windows; `vgamepad` needs the ViGEmBus
  driver). Nothing in the environment layer is portable today.
- NVIDIA GPU with CUDA 12.x for any training run of practical length.
- Python 3.11 (the frozen environment targets `cp311`).

### 13.2 Install

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is a verbatim `pip freeze` of the working environment, including
`torch==2.5.1+cu121`, which resolves only from PyTorch's CUDA index:

```powershell
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

For a fresh, minimal install instead of the frozen file, see
[docs/SETUP.md §5](docs/SETUP.md#5-install-path-b--curated-minimum), which lists the exact set derived
from the real import statements. Abbreviated:

```powershell
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install gymnasium==0.29.1 stable-baselines3==2.2.1 imitation==1.0.1 mlflow \
            numpy==2.4.4 opencv-python psutil pygame pywin32 vgamepad dxcam comtypes \
            keyboard mss PyDirectInput mouse
```

### 13.3 Verify the input path

```powershell
python test_xinput.py          # lists connected XInput controllers, if any
Get-Service ViGEmBus           # must exist for vgamepad to work at all
```

`vgamepad` is only a Python binding; the **ViGEmBus kernel driver must be installed separately**. If it
is absent, `import vgamepad` raises at module load — which, because `game_env.py` imports it at module
scope, makes the environment module unimportable rather than merely non-functional. On the machine this
revision was written on, `Get-Service ViGEmBus` reports the service absent, so no recording or deployment
command can currently be executed there.

### 13.4 Docker

`Dockerfile` is provided for CUDA training but **does not currently work**: it builds from
`nvidia/cuda:12.1.1-runtime-ubuntu22.04`, which ships no Python interpreter, and its `CMD` invokes a
training script. In addition, the `sed` line that strips Windows-only dependencies leaves
`pydirectinput` and `mouse` uninstalled while `game_env.py` imports `vgamepad` at module scope, so
the environment module cannot be imported on Linux at all. Treat the file as a statement of intent.

---

## 14. Usage guide

The full operator-facing procedure — including recording hygiene, what a healthy loss curve looks
like, and how to run DAgger corrections — is in [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md).
Condensed:

```bash
cd generic_agent/notebooks        # or hajime_agent/notebooks

# 1. Launch the game/emulator first, then record. K = start/stop (K is what SAVES), ESC = quit.
python record_trajectories.py

# 2. Behavioural cloning. K = toggle AI/manual, ESC = quit.
python train_agent.py --epochs 100 --batch 384 --lr 1e-4

# 3. Inspect the live loss curves.
mlflow ui --backend-store-uri file:../mlruns      # http://localhost:5000

# 4. Deploy.
python run_ai.py                                  # or run_ai_lstm.py / run_ai_transformer.py

# 5. Correct the agent's mistakes (L hands control back to you), then retrain.
python run_dagger.py
python train_agent.py                             # retrains over demos/ as a whole

# 6. Re-run the architecture study (regenerates §10.1 and comparison_results.md).
python compare_models.py --epochs 10 --batch 384 --lr 1e-4
```

Recording practice that the results in §10 depend on: **variety beats volume.** Short clips
(1–2 min) covering distinct situations — offence, defence, whiffed combos, being cornered — produce a
usable action distribution; long unbroken sessions do not. The corpus statistics in §6.2 show what
happens otherwise: 11 of 18 action bits never fired.

---

## 15. Reproducibility

### 15.1 Determinism

`compare_models.py` seeds a `numpy` Generator with 42 for demonstration shuffling. No `torch.manual_seed`,
`cudnn.deterministic`, or `torch.use_deterministic_algorithms` call exists anywhere, so CUDA kernel
nondeterminism is uncontrolled — which §10.2 shows is material at this sample size. Nothing pins the
Python or CUDA build beyond `requirements.txt`.

### 15.2 What is and is not versioned

| Artefact | Path | In git? |
|---|---|---|
| Source code, docs, config | `*/utils`, `*/notebooks/*.py`, `*/config`, `docs/` | yes |
| Reference BC policy | `generic_agent/notebooks/models/bc_policy.zip` (17 MB) | yes |
| Generated comparison report | `generic_agent/notebooks/models/comparison_results.md` | yes |
| One historical training log | `generic_agent/notebooks/models/imitation/bc_logs/progress.csv` | yes |
| Human demonstrations | `*/notebooks/demos/*.pt` (~3.6 GB) | **no** (`*.pt`) |
| Trained checkpoints | `*/notebooks/models/*_policy.zip` | **no** (`*.zip`) |
| MLflow store (all 31 runs) | `*/mlruns/` | **no** |

The `.gitignore` was rewritten in this revision to be explicit about that boundary and to keep the
reference policy tracked via a negation rule. Consequences: the study's evidence base is currently
single-machine, and a 17 MB binary sits in git history without Git LFS configured (`git lfs init`/
`install` has never been run in this clone). Migrating `bc_policy.zip` to LFS or to a release asset is
the recommended next step (§12.7).

---

## 16. Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `Window not found` / `Waiting for game window…` | Game not running, or `process_name` does not match | Start the title first; set `process_name` to the exact substring of the owning process |
| DXCam warning, falls back to `mss` | Dual-GPU laptop; duplication attaches to the adapter that is not presenting the window | Expected on Optimus laptops; capture is slower — raise `internal_*` reduction or accept lower FPS |
| Captured region is offset / includes the title bar | Window decorations included in the grab rectangle | Tune `window_offset.{left,top,right,bottom}` |
| `import vgamepad` raises a `VIGEM_ERRORS` name at start-up | ViGEmBus driver not installed (the pip package does not install it) | Install ViGEmBus, reboot, confirm with `Get-Service ViGEmBus`, then re-run `python test_xinput.py` |
| Virtual pad exists but the game ignores it | The title was started before the pad was attached, or polls DirectInput only | Start the game *after* the environment creates the pad |
| Training loss plateaus around 3.0 and never improves | 10-epoch budget on a 27 k-frame corpus | Train far longer; §10.3 reaches 1.33 loss at ~290 epochs |
| Policy only ever presses two directions | Action marginals dominate (§6.2) and the corpus lacks variety | Record the missing actions explicitly; consider label re-weighting |
| `CUDA out of memory` | Batch 384 over 4×128×128 float, larger on flatten heads | Lower `--batch`, or prefer the GAP-based encoder |
| `ModuleNotFoundError: vgamepad` / `pywin32` | Non-Windows platform | Not supported; see §13.4 |
| `train_imiation.py` reports zero demonstrations | It globs `demos*.pt`, every other script writes `demo*.pt` | Use `train_agent.py`, or fix the glob |

---

## 17. Documentation index

| Document | Covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Environment internals, capture/actuation paths, every feature extractor with tensor shapes, module dependency graph |
| [`docs/DATA.md`](docs/DATA.md) | Recording protocol, demonstration file format, corpus statistics, action-space integrity |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | Protocol per script, CLI reference, MLflow metric definitions, how to add an architecture |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Full run-by-run results from the tracking store, dispersion analysis, withdrawn architectures |
| [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) | Threats to validity expanded, plus a prioritised open-work list |
| [`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md) | Operator-facing step-by-step guide (English successor to the removed `GUIA_TREINAMENTO.md`) |
| [`docs/SETUP.md`](docs/SETUP.md) | Environment installation, Windows-only dependency map, Docker status |

---

## 18. License and attribution

The environment design, the recorder/trainer loop and the demonstration format follow the work
published by [@paulo101977](https://github.com/paulo101977/notebooks-rl), from which this project was
derived; `GenericGameEnv` is an adaptation of that author's `resident_requiem.py`. The
`Impoola-CNN` encoder, the ViT extractor, the benchmark harness and the MLflow instrumentation are
original to this repository.

**No licence file is present in the repository.** Licensing has not been chosen, so no usage rights
are granted beyond those implied by GitHub's terms for a public repository. Resolving this is listed
as open work in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).
