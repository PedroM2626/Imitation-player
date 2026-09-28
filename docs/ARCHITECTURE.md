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
   operator's workstation. This favours a shared `BaseFeaturesExtractor` seam behind a single
   architecture registry (`agent/utils/architectures.py`) and one `--arch`-selectable entry point.

## 2. Module map

```
                        +-----------------------------------+
                        | agent/config/profiles/*.py        |  plain dicts, no schema
                        |  GAME_CONFIG / TRAINING_CONFIG    |
                        |  (+ agent/config/local.py,        |  machine-specific overrides
                        |   git-ignored, from local.example)|
                        +----------------+------------------+
                                         | read by agent/config/__init__.py (load_profile)
          +------------------------------+-------------------------------+
          |                                                              |
+---------v-----------+                                        +---------v-----------+
| utils/game_env.py   |  GenericGameEnv(gym.Env)               | agent/cli/*.py      |
|                     |  + WindowNotFoundError                 |  pipelines          |
+---------+-----------+                                        +---------------------+
          |                                                          |
          |                                                   imports
          |                          +---------------+----------------+---------------+
          |                          |               |                |               |
          v                          v               v                v               v
   utils/windows.py           utils/architectures.py  utils/demos.py   utils/emission.py
   HAS_WIN32 / HAS_DXCAM /    ARCHITECTURES,         load_demos,      GamepadEmitter,
   HAS_VGAMEPAD / HAS_MSS     Architecture,          summarise,       KeyboardMouseEmitter,
   guard dxcam, mss,          policy_kwargs_for      print_summary,   validate_mappings
   win32gui, win32process                            DemoError
   and vgamepad               utils/checkpoints.py           |
                              get_last_index,                |
   utils/paths.py             list_checkpoints,             imports
   runs_root, demos_dir,      resolve_checkpoint             v
   models_dir, mlruns_dir,                                 utils/temporal_lstm.py
   logs_dir                    utils/tracking.py           TemporalAttentionLSTM
                              MetricCapture,
   utils/utils.py             MLflowOutputFormat,          utils/vision_transformer.py
   PolicyRunner, load_policy  start_run, log_dataset       PatchEmbedding,
                                                           TransformerEncoderBlock,
                              utils/input_map.py           VisionTransformerExtractor
                              HumanInput, XInputState
                                                           utils/impoola_cnn.py
                                                           ResidualBlock, ImpoolaBlock,
                                                           ImpoolaCNNExtractor
                                                           utils/new_architectures.py
                                                           ImpalaCNNExtractor,
                                                           ResNet18Extractor
```

One package, `agent/`, holds this graph. The two near-duplicate packages this graph was split from have
been collapsed into it, with the per-title differences now expressed as two profiles in
`agent/config/profiles/` (`hajime_ippo`, `roblox`) that still differ in input mode and action width.

## 3. `GenericGameEnv`

`agent/utils/game_env.py` — the only class in the project that touches the operating system.

### 3.1 Construction sequence

```
__init__(config)
 ├─ resolve config (fall back to `agent.config.game_config()`, i.e. the default profile, if none is passed)
 ├─ read capture geometry (internal_width/height, target_fps, buffer_len)
 ├─ action_space = MultiBinary(actions.num_actions)
 ├─ observation_space = Box(0, 255, (internal_h, internal_w, 1), uint8)
 ├─ if dummy: return early            <-- window / emitter / camera are never created
 ├─ hwnd = find_window_by_process_name(process_name)
 ├─ if not hwnd and exe_path: Popen([exe_path, rom_path])
 ├─ wait_start()                      <-- polls for the window for up to 120 s, then raises
 ├─ emitter = emission.build_emitter(config, windows.vg)
 │    ├─ "gamepad":        vg.VX360Gamepad(), driven by actions.mappings
 │    └─ "keyboard_mouse": import pydirectinput; FAILSAFE = False
 └─ camera
      ├─ dxcam.create(output_color="GRAY", max_buffer_len=buffer_len)
      │    .start(region=_window_region(), target_fps=target_fps)
      └─ on any exception: mss.mss()    (CPU fallback; prints which backend was unavailable)
```

Notable properties:

- Construction **blocks up to two minutes** waiting for a window and, on timeout, raises
  `WindowNotFoundError` rather than returning an object whose every later capture is the zero/black
  fallback. The CLI wrappers turn that into a one-line message instead of a traceback
  (`agent.cli.common.cli_entry`).
- `target_fps` now defaults to 60 in the environment's own `.get` fallback as well as in both profiles.
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
| Failure | `frame is None` → return `self.img`, else `np.zeros((H, W, 1), uint8)` | `self.dropped_frames` now counts these, but no exception or per-frame warning is raised, so the stored observation is still indistinguishable from a legitimately black game frame |
| Resize | `cv2.resize(..., (internal_w, internal_h), INTER_NEAREST)` | nearest-neighbour, deliberately: cheaper, and anti-aliasing would blur HUD text into illegibility |
| Channel | `np.expand_dims(..., -1)` if the result is 2-D | keeps the declared `(H, W, 1)` shape |

`INTER_NEAREST` means the 128×128 image is a subsample, not a resample: at a typical 1920×1080 source
only ~0.7 % of pixels survive, so fine HUD text and thin health-bar edges alias. This is the concrete
mechanism behind the "spatial precision" argument in README §7.4 — it is a capture-time loss, not a
GAP-induced one.

### 3.4 Action emission (`step`)

Two independent emitters share the `MultiBinary` vector, both living in
`agent/utils/emission.py` and both driven by the profile's `actions.mappings` table.

**Gamepad path** (`input_mode == "gamepad"`). `GamepadEmitter` resolves the active bits against the
table: each mapping declares a `kind` (`button`, `axis` or `trigger`) plus `button` / `stick` / `axis` /
`value` / `trigger`, and `button` names are looked up in the `XBOX_BUTTON` table, so the profile's names
are the same names the emulator's XInput layer sees — the former `vg_code` / `DS4_BUTTON_*` mismatch no
longer exists. The resolved state is then diffed against the previously held set:

```
for b in pressed - held:  press_button(b)              # discrete only
for b in held - pressed:  release_button(b)
left_joystick_float / right_joystick_float / left_trigger / right_trigger
gamepad.update()             # commit the report to the virtual device
held = pressed
```

Buttons are written only on transition, but axes and triggers are re-written on every `apply()` call, so
a released stick is actively re-centred instead of keeping its last value. Conflicting bits on one axis
now **sum and clamp** to [-1, 1] (`max(-1.0, min(1.0, axes[key] + value))`) rather than the previous
last-index-wins (`if actions[10]: ... ; if actions[11]: ...`), so a contradictory prediction resolves to
the algebraic combination of the flags.

**Keyboard/mouse path** (`input_mode == "keyboard_mouse"`). `KeyboardMouseEmitter` iterates the same
table, adds indices whose bit is set to the held set, and issues `keyDown/keyUp` or `mouseDown/mouseUp`
on the symmetric difference. This mode is now simply the other branch of the same mechanism — the
`actions.mappings` key is honoured identically on both input modes, and both paths fall through to the
same frame-budget pacing at the end of `step()` — the remaining `1 / capture.target_fps` seconds are
slept out, so `target_fps` is no longer a field that capture alone obeys.

Both branches return `(observation, 0.0, False, False, {})`. Before either runs, `step()` validates the
vector against `actions.num_actions` and raises `ValueError` on a width mismatch (the old code indexed
`actions[10:18]` unconditionally, so a narrower gamepad configuration simply crashed later), and it
refuses to actuate at all from a `dummy` environment.

### 3.5 Wrappers actually used

```python
env = GenericGameEnv(cfg)          # cfg["dummy"] = True for training
env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)       # (H, W, 1) -> (1, H, W)
env = VecFrameStack(env, n_stack=4)  # (1, H, W) -> (4, H, W)
```

Applied by `agent.cli.common.wrapped_env`, which every script that needs an environment calls
(`agent/cli/train.py`, `agent/cli/benchmark.py`, `agent/cli/record.py`, `agent/cli/deploy.py`,
`agent/cli/dagger.py`, `agent/cli/train_gail.py`). Absent wrappers worth naming: `OrderEnforcing`,
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

### 4.1 `ImpoolaCNNExtractor` — `agent/utils/impoola_cnn.py`

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

### 4.2 `ImpalaCNNExtractor` — `agent/utils/new_architectures.py:50`

Identical convolutional stages, terminating in
`Flatten → Linear(32768 → 512)`. The flatten dimension is derived from the pooled spatial extent
(`16 × 16 × 128`), so it is valid only when `H = W` and divisible by 8; a non-square or
non-power-of-two capture resolution raises at construction rather than at forward time.

### 4.3 `ResNet18Extractor` — `agent/utils/new_architectures.py:98`

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

### 4.4 `VisionTransformerExtractor` — `agent/utils/vision_transformer.py`

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

### 4.5 `TemporalAttentionLSTM` — `agent/utils/temporal_lstm.py:28`

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
- `window_size` is now a constructor argument, and `agent/utils/architectures.policy_kwargs_for` passes
  `TRAINING_CONFIG["window_size"]` (default 10, the previous hard-coded value) into it, so the config
  field is no longer decorative.
- `SpatialAttention` was dead code with no importer; it has been deleted. `TemporalAttentionLSTM` moved
  out of `game_env.py` into `agent/utils/temporal_lstm.py` with its numerics unchanged — still stateful,
  still `requires_grad_(True)` on the stacked sequence — so the published CNN+LSTM benchmark row stays
  comparable.

## 5. Inference path (`agent/cli/deploy.py`)

```
resolve checkpoint -> load_policy -> policy.predict(obs) -> threshold to bits -> env.step(bits)
```

`agent/cli/deploy.py` contains an **aggressiveness** alternative to `policy.predict`: when the factor is
not `1.0` (and the profile names at least one attack button), `PolicyRunner.sharpened`
(`agent/utils/utils.py:26`) runs `extract_features → mlp_extractor → action_net`, sigmoids the logits,
multiplies the probability of every index whose `mappings` entry has a `name` listed in
`deploy.attack_buttons` by the factor, clamps to 1.0, and then samples with
`np.random.rand() < probs`. It reads the factor from `GAME_CONFIG["deploy"]["aggressiveness"]`
(`agent/cli/deploy.py:51`), the same dictionary the profile defines it in, so the branch is reachable by
editing the profile rather than the script; `deploy.fps` is honoured in the same loop. Note also that
the sampling path bypasses SB3's own action masking/dtype handling, so it is not equivalent to
`predict(deterministic=False)`.

`PolicyRunner` (`agent/utils/utils.py:11`, replacing the old `LSTMWrapper`) routes recurrent models
through `predict` and calls the extractor's `reset_hidden()` when it is reset. The deploy loop calls
`runner.reset()` when control is toggled with `K`; since `done` is never `True`
([README §5.3](../README.md#53-reward-termination-and-the-deployment-loop)), no step ever triggers that
reset by itself.

`get_last_index(dir, prefix, ext)` (`agent/utils/checkpoints.py:16`) resolves SB3's `bc_policy<N>.zip`
numbering by matching `prefix + r"(\d+)" + ext` and taking `max(int(...))`, returning `-1` when nothing
matches; the comparison is numeric, and `list_checkpoints` sorts the same way. `resolve_checkpoint()`
(`agent/utils/checkpoints.py:41`) then prefers the final `<prefix>.zip` and falls back to the highest
numbered checkpoint, both rooted at the profile's `runs/<profile>/models/` directory. The dead
`STEPS_PATH = "./models/steps"` the old deployment scripts searched — a directory no script ever
created, which made the numbered fallback unreachable and pinned deployment to `models/bc_policy.zip` —
is gone.

## 6. Tracking and artefacts

| Concern | Mechanism | Location |
|---|---|---|
| Experiment store | MLflow file backend `file:runs/<profile>/mlruns` (`agent/utils/paths.mlruns_dir`, overridable with `IMITATION_RUNS` / `--runs-root`) | `runs/<profile>/mlruns/<exp_id>/<run_id>/` |
| Experiment names | historical: `Model_Comparison`, `Hajime_no_Ippo_Imitation_Learning`; new runs use `f"{profile}_{purpose}"` (`hajime_ippo_imitation_bc`, `hajime_ippo_gail`) | 26 and 5 runs respectively in the retained store |
| Per-batch metrics | `MLflowOutputFormat(prefix=f"{arch.name}/")` from `agent/utils/tracking.py` | `metrics/<Model>/bc/*`; `bc/l2_loss` is withheld (`DEFAULT_SKIP_KEYS`), `bc/l2_norm` is not |
| Summary metrics | explicit `mlflow.log_metric`, plus `tracking.log_dataset()` on every training run | `final_loss`, `training_time_s`, `num_params`, `model_size_mb`; and the corpus frame/trajectory counts, action width, distinct joint-action count, `marginal_baseline_nats` and `uniform_baseline_nats` |
| Ad-hoc capture | `MetricCapture` Logger sink in `agent/utils/tracking.py`, installed by `agent/cli/train.py` | reads the last `bc/loss` value |
| Checkpoints | `policy.save()` | `runs/<profile>/models/<Name>_policy.zip` |
| Generated reports | `benchmark.regenerate_readme()` / `benchmark.write_report()` | rewrites `README.md` between `<!-- BENCHMARK_START -->` / `<!-- BENCHMARK_END -->` and regenerates `runs/<profile>/models/comparison_results.md` |

The README-injection behaviour deserves care: **running `python -m agent.cli.benchmark` rewrites the
top-level README.** It replaces the marked table and rewrites the sentence following the
`**Conclusion**:` label — which it now generates from the measured values rather than from a
hand-written string; it does not touch anything else. Keep the marker comments intact when
editing README.

## 7. Known architectural debt

Ordered by the cost they impose on the research:

1. **No evaluation seam.** Nothing in the design forces a split or a rollout; adding evaluation means
   adding a reward/terminal signal, which is an environment change, not a script change.
2. **Stateful extractor.** `TemporalAttentionLSTM`'s internal buffer and hidden state make its output
   a function of call history, breaking the batch-independence assumption the other encoders satisfy.
3. **One package, two profiles.** The two-package fork — ten modules duplicated byte-for-byte, with
   benchmark baselines that already disagreed between copies — has been removed: both copies collapse
   into `agent/`, and the per-title differences live in `agent/config/profiles/`. What remains is that
   the two profiles still differ in input mode and action width, so their losses are not comparable.
4. **Config was advisory, and mostly is not any more.** `actions.mappings` now drives gamepad emission as
   well as keyboard/mouse, is validated against `actions.num_actions` at load time, and the dead fields
   (`capture.width`/`height`, `demo_path`/`model_path`/`train_path`, `INPUT_CONFIG` as a separate
   dictionary) have been removed or flattened into `GAME_CONFIG`. `TRAINING_CONFIG.window_size`,
   `.dagger_iterations` and `recording.max_trajectories` are consumed.
5. **Failure was silent, and now mostly is not.** Window lookup raises `WindowNotFoundError`, a
   width-disagreeing demonstration raises `DemoError` under the default `strict` policy, and an
   unreadable demo file is a hard error listing every failure. The residue is capture: a dropped grab is
   counted (`self.dropped_frames`) but still substitutes the previous or a black frame, and the counter is
   not written into the demonstrations. In a study with no held-out metric, an unremarked data-integrity
   failure is indistinguishable from a result.
6. **`Dockerfile` is repaired but unverified, and `venv/` is still dead weight** — see [SETUP.md](SETUP.md).
