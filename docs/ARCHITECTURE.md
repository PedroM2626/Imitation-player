# Architecture

> Deep dive into the runtime design of Imitation Player. Overview and results live in the
> [top-level README](../README.md); measurement details live in [RESULTS.md](RESULTS.md).

## 1. Design constraints

Three constraints determine almost every design decision below.

1. **Black-box access.** The agent may not read game memory, inject engine calls, or use an SDK. Its
   only sensor is the framebuffer; its only actuator is a synthesised input device. Consequently the
   observation is pixels the GPU was never asked to hand over, and the action is an event the OS was
   never asked to synthesise — both require platform-specific side channels.
2. **No task signal.** Nothing in a third-party game tells you whether the player is winning. Without
   a reward, reinforcement learning has no objective; the project therefore uses supervised
   imitation, which needs only `(state, action)` pairs. The cost is that the environment's reward and
   terminal channels are left unimplemented (§3.4, [README §5.3](../README.md#53-reward-termination-and-the-deployment-loop)),
   which in turn forecloses every episodic metric.
3. **Human-in-the-loop research cadence.** The system exists to compare encoders quickly on one
   operator's workstation. This favours a shared `BaseFeaturesExtractor` seam with per-architecture
   entry-point scripts over an abstracted model registry.

## 2. Module map

```
                        +-----------------------------------+
                        | config/game_config.py             |  plain dicts, no schema
                        |  GAME_CONFIG / TRAINING_CONFIG /  |
                        |  INPUT_CONFIG                     |
                        +----------------+------------------+
                                         | read by
          +------------------------------+-------------------------------+
          |                                                              |
+---------v-----------+                                        +---------v-----------+
| utils/game_env.py   |  GenericGameEnv(gym.Env)               | notebooks/*.py      |
|                     |  + SpatialAttention (orphaned)         |  pipelines          |
|                     |  + TemporalAttentionLSTM               +---------------------+
+---------+-----------+                                                  |
          |                                                        imports
          |                                    +------------------------+------------------+
          |                                    |                        |                  |
          v                                    v                        v                  v
   dxcam / mss / win32gui               utils/impoola_cnn.py     utils/             utils/vision_
   vgamepad / pydirectinput             utils/new_architectures.py  vision_transformer.py
                                         (ResidualBlock,            (PatchEmbedding,
                                          ImpoolaBlock,              TransformerEncoderBlock,
                                          ImpalaCNNExtractor,        VisionTransformerExtractor)
                                          ResNet18Extractor)

   utils/utils.py — get_last_index(dir, prefix, ext), LSTMWrapper(model)
```

Both packages (`generic_agent`, `hajime_agent`) contain this graph. Nine modules are byte-identical
between them; the divergences are listed in [README §3.1](../README.md#31-the-two-package-fork).

## 3. `GenericGameEnv`

`utils/game_env.py` — the only class in the project that touches the operating system.

### 3.1 Construction sequence

```
__init__(config)
 ├─ resolve config (fall back to an embedded default dict if `config.game_config` is unimportable)
 ├─ read capture geometry (width/height, internal_width/height, target_fps, buffer_len)
 ├─ action_space = MultiBinary(actions.num_actions)
 ├─ observation_space = Box(0, 255, (internal_h, internal_w, 1), uint8)
 ├─ if dummy: return early            <-- window / pad / camera are never created
 ├─ hwnd = find_window_by_process_name(process_name)
 ├─ if not hwnd and exe_path: Popen([exe_path, rom_path])
 ├─ wait_start()                      <-- polls for the window for up to 120 s
 ├─ input controller
 │    ├─ "gamepad":        vg.VX360Gamepad()
 │    └─ "keyboard_mouse": import pydirectinput; FAILSAFE = False   (generic_agent only)
 └─ camera
      ├─ dxcam.create(output_color="GRAY", max_buffer_len=buffer_len)
      │    .start(region=_get_window_region(), target_fps=target_fps)
      └─ on any exception: mss.mss()    (CPU fallback; prints a warning block)
```

Notable properties:

- Construction **blocks up to two minutes** waiting for a window, and on timeout it only prints a
  warning: the object remains usable in a permanently broken state (every subsequent capture returns
  the zero/black fallback). Fail-fast would be more honest here.
- `target_fps` defaults to 240 in the embedded fallback dict but 60 in both real configs.
- `hide_window` moves the window to `(-w, -h)`, i.e. off-screen. The capture region is then computed
  from that same off-screen rectangle, so the grabbed region is off-monitor. With Desktop Duplication
  the behaviour is adapter-dependent and not characterised in this repository.

### 3.2 Window discovery

`find_window_by_process_name` enumerates all visible top-level windows with
`win32gui.EnumWindows`, resolves each window's PID via `GetWindowThreadProcessId`, reads the process
name through `psutil`, and returns the **first** window whose process name contains the configured
substring, case-insensitively. Multiple candidate windows (an emulator with several top-level windows,
a matching IDE process) are not disambiguated — selection is enumeration order, which is z-order
dependent.

### 3.3 Capture and preprocessing (`_get_observation`)

| Step | Implementation | Notes |
|---|---|---|
| Grab | `camera.get_latest_frame()` (dxcam) or `sct.grab(monitor)` → `cv2.cvtColor(BGRA2GRAY)` (mss) | dxcam already returns grayscale; mss converts |
| Failure | `frame is None` → return `self.img`, else `np.zeros((H, W, 1), uint8)` | **silent** — no counter, no warning, indistinguishable from a legitimately black game frame |
| Resize | `cv2.resize(..., (internal_w, internal_h), INTER_NEAREST)` | nearest-neighbour, deliberately: cheaper, and anti-aliasing would blur HUD text into illegibility |
| Channel | `np.expand_dims(..., -1)` if the result is 2-D | keeps the declared `(H, W, 1)` shape |

`INTER_NEAREST` means the 128×128 image is a subsample, not a resample: at a typical 1920×1080 source
only ~0.7 % of pixels survive, so fine HUD text and thin health-bar edges alias. This is the concrete
mechanism behind the "spatial precision" argument in README §7.4 — it is a capture-time loss, not a
GAP-induced one.

### 3.4 Action emission (`step`)

Two independent emitters share the `MultiBinary` vector.

**Gamepad path** (`input_mode == "gamepad"`, both packages). Indices are hard-coded (§README 5.2). The
method builds a `current` set of symbolic button ids — real `XUSB_BUTTON` flags for discrete buttons
and negative sentinels for continuous channels (`-2` left stick, `-3` right stick, `-4` left trigger,
`-5` right trigger) — then diffs it against `self.prev_keys`:

```
for b in current - prev_keys:  press_button(b)              # discrete only
for b in prev_keys - current:  release_button(b) / zero the axis or trigger
left_joystick_float / right_joystick_float / left_trigger / right_trigger
gamepad.update()             # commit the report to the virtual device
prev_keys = current
```

Re-emitting an unchanged axis every step is skipped by design (axes are only written on transition),
which means a *held* stick keeps its last value — correct — but also means the axis is never
re-centred unless the corresponding bit turns off. The last `right_joystick_float` value for a group
of two opposing flags wins (`if actions[10]: ... ; if actions[11]: ...`), so a contradictory
prediction resolves to the higher index rather than to a cancellation or an error.

**Keyboard/mouse path** (`generic_agent` only). Iterates `self.mappings`, adds indices whose bit is
set to `current`, and issues `keyDown/keyUp` or `mouseDown/mouseUp` on the symmetric difference with
`prev_keys`. Note this branch **does** read the config table, unlike the gamepad branch — so the same
config key (`actions.mappings`) is honoured on one input mode and ignored on the other. The branch
returns early, before any gamepad code, and therefore also before any `frame_time` pacing.

Both branches return `(observation, 0.0, False, False, {})`.

### 3.5 Wrappers actually used

```python
env = GenericGameEnv(cfg)          # cfg["dummy"] = True for training
env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)       # (H, W, 1) -> (1, H, W)
env = VecFrameStack(env, n_stack=4)  # (1, H, W) -> (4, H, W)
```

Applied in `train_agent*.py` and `compare_models.py`. Absent wrappers worth naming: `OrderEnforcing`,
`ClipAction` (unnecessary for `MultiBinary`), `TransformObservers` (normalisation is done inside each
extractor instead), and `TimeLimit` (so episodes are unbounded).

Because frame stacking happens twice — once in the recorder (demos are stored as `(N, 4, 128, 128)`)
and once via `VecFrameStack` at training time — the stacked tensor the encoder receives during BC is
built from already-stacked demonstrations for the supervised path, while the environment path stacks
single frames. The two are consistent in *shape* but not produced by the same code, which is a
train/deploy skew a held-out evaluation would have caught (§README 12.1).

## 4. Feature extractors

All five encoder modules obey the same contract:

```python
class SomeExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, ...):
        super().__init__(observation_space, features_dim)
    def forward(self, observations: Tensor) -> Tensor: ...   # (B, C, H, W) -> (B, features_dim)
```

`features_dim` then feeds `ActorCriticCnnPolicy`'s `mlp_extractor` and its `action_net`, which is
`Linear(features_dim -> num_actions)` producing independent sigmoid logits per bit.

### 4.1 `ImpoolaCNNExtractor` — `utils/impoola_cnn.py`

```
n_obs = observation_space.shape          # expects (4, 128, 128)
Conv2d(4→32, 3×3) ReLU MaxPool2d(2)      # 128² → 64²
Conv2d(32→64, 3×3) ReLU MaxPool2d(2)     # 64² → 32²
Conv2d(64→128, 3×3) ReLU MaxPool2d(2)    # 32² → 16²
   └ each stage is followed by two ResidualBlock(3×3 conv + skip)
AdaptiveAvgPool2d((1,1))                 # 16×16×128 → 1×1×128   <-- GAP
Flatten -> Linear(128→512)               # 128 params in, not 32,768
```

Normalises internally when the input is unsigned (`/255` if `max > 1`). The channel schedule
`[32, 64, 128]` and `features_dim=512` are constructor defaults and are the values used in the
benchmark.

### 4.2 `ImpalaCNNExtractor` — `utils/new_architectures.py:50`

Identical convolutional stages, terminating in
`Flatten → Linear(32768 → 512)`. The flatten dimension is derived from the pooled spatial extent
(`16 × 16 × 128`), so it is valid only when `H = W` and divisible by 8; a non-square or
non-power-of-two capture resolution raises at construction rather than at forward time.

### 4.3 `ResNet18Extractor` — `utils/new_architectures.py:98`

```
torchvision.resnet18(weights=None)
  model.conv1 = Conv2d(4, 64, 7, stride=2, padding=3, bias=False)   # accept stacked frames
  model.fc    = Identity()                                          # emits 512-d
  head        = Linear(512 → features_dim)
```

The stem's 7×7/s2 kernel is retained from ImageNet practice. At 128×128 input that yields a 64×64
activation map immediately, which is generous for a grayscale screen capture and is the main reason
ResNet-18's early-epoch loss is the lowest in the study — the stem preserves more of the frame than
the 3×3/s1-then-pool stems of the Impala-family encoders do, at a fraction of the parameter cost
because no flatten head follows.

### 4.4 `VisionTransformerExtractor` — `utils/vision_transformer.py`

```
PatchEmbedding: Conv2d(4 → 256, kernel=stride=16)      # (4,128,128) -> 8×8 grid of 256-d tokens
tokens = [CLS; flatten(patches)]                        # 1 + 64*(4 frames) = 257
+ spatial_pos_emb[256]                                  # per patch position
+ temporal_emb[4]                                       # per source frame, broadcast over patches
4 × TransformerEncoderBlock:
    LayerNorm → MultiheadAttention(4 heads) → residual
    LayerNorm → FFN(256 → 512 → 256), GELU → residual   # mlp_ratio=2.0, dropout 0.1
LayerNorm → CLS token → Linear(256 → features_dim)
```

Pre-normalisation blocks (LN before attention), a single learnable class token, and *explicit*
temporal embeddings are the three departures from vanilla ViT; without the last one the model could
not distinguish frame *t* from *t−1*, since patch tokens are otherwise permutation-equivariant within
the stack.

### 4.5 `TemporalAttentionLSTM` — `utils/game_env.py:379`

The only extractor that is stateful. Full description in [README §7.2](../README.md#72-cnn--lstm--temporal-attention-temporalattentionlstm-utilsgame_envpy379-523).

Architectural specifics worth recording:

- The CNN treats the 4 stacked frames as *input channels*, not as a time axis, and collapses space to
  `AdaptiveAvgPool2d(1,1)` — so the LSTM's 512-d per-step vectors are global averages with no layout.
- Gradients are cut on buffer entries (`detach()`), then the stacked sequence is re-flagged
  `requires_grad_(True)` in place. The net effect is that **only the most recent timestep
  back-propagates through the CNN**; earlier positions train the LSTM and attention only. This is not
  documented in the code and materially changes what "trains an LSTM" means.
- Hidden state carries across `forward()` calls via `self.hidden_state`, and `repackage_hidden`
  detaches it. Since BC minibatches are shuffled and no episode boundary exists, the carried state
  mixes unrelated sequences. `reset_hidden()` is never called by any training loop.
- `window_size = 10` is hard-coded; `TRAINING_CONFIG["window_size"]` is unused.
- `SpatialAttention` (`game_env.py:368`) is dead code: no importer references it.

## 5. Inference path (`run_ai*.py`)

```
load zip -> policy.predict(obs) -> threshold to bits -> env.step(bits)
```

`run_ai.py` (generic) adds an optional **aggressiveness** term: instead of sampling, it applies
`action_net` to the features, passes through a sigmoid, and raises the probabilities to a power from
`INPUT_CONFIG["aggressiveness"]` before thresholding. This sharpens the argmax distribution at the
cost of suppressing low-confidence actions — an inference-time bias with no evaluation measuring its
effect. `LSTMWrapper` (`utils/utils.py:42`) exists to route recurrent models through `predict` while
calling `reset_hidden()` on done flags, but since `done` is never `True`
([README §5.3](../README.md#53-reward-termination-and-the-deployment-loop)), that reset never fires during deployment either.

`get_last_index(dir, prefix, ext)` resolves SB3's `bc_policy<N>.zip` checkpoint numbering by string
parsing to pick the highest *N*. It compares lexicographically rather than numerically, so
`bc_policy9.zip` outranks `bc_policy10.zip`.

## 6. Tracking and artefacts

| Concern | Mechanism | Location |
|---|---|---|
| Experiment store | MLflow file backend `file:../mlruns` | `*/mlruns/<exp_id>/<run_id>/` |
| Experiment names | `Model_Comparison`, `Hajime_no_Ippo_Imitation_Learning` | 26 and 5 runs respectively |
| Per-batch metrics | `MLflowOutputFormat(prefix="<Model>/")` from `imitation` | `metrics/<Model>/bc/*` |
| Summary metrics | explicit `mlflow.log_metric` | `final_loss`, `training_time_s`, `num_params`, `model_size_mb` |
| Ad-hoc capture | `MetricCapture` Logger sink in `compare_models.py` | reads the last `bc/loss` value |
| Checkpoints | `policy.save()` | `models/<Name>_policy.zip` |
| Generated reports | `save_and_update_results()` | rewrites `README.md` between `<!-- BENCHMARK_START -->` / `<!-- BENCHMARK_END -->` and regenerates `models/comparison_results.md` |

The README-injection behaviour deserves care: **running `compare_models.py` rewrites this
document.** It replaces the marked table and rewrites the sentence following the
`**Conclusion**:` label; it does not touch anything else. Keep the marker comments intact when
editing README.

## 7. Known architectural debt

Ordered by the cost they impose on the research:

1. **No evaluation seam.** Nothing in the design forces a split or a rollout; adding evaluation means
   adding a reward/terminal signal, which is an environment change, not a script change.
2. **Stateful extractor.** `TemporalAttentionLSTM`'s internal buffer and hidden state make its output
   a function of call history, breaking the batch-independence assumption the other encoders satisfy.
3. **Two packages, one code base.** Nine byte-identical modules duplicated; the benchmark's hard-coded
   baselines already disagree between copies.
4. **Config is advisory.** Several declared fields are read by nothing, and the field that matters most
   in gamepad mode (`actions.mappings`) is ignored.
5. **Silent failure everywhere.** Capture, demo loading, window lookup and action-width reconciliation
   all degrade without raising. In a study with no held-out metric, an unremarked data-integrity
   failure is indistinguishable from a result.
6. **`Dockerfile` and `venv/` are both non-functional** — see [SETUP.md](SETUP.md).
