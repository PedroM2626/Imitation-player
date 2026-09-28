# Training Guide

> *The English successor to `GUIA_TREINAMENTO.md`, which the previous README referenced and which no
> longer exists in the tree.* Operator-facing and in pipeline order; rationale lives in
> [ARCHITECTURE.md](ARCHITECTURE.md), installation in [SETUP.md](SETUP.md), measurements in
> [RESULTS.md](RESULTS.md). All commands assume an active Python 3.11 virtual environment and are run as
> module invocations from the **repository root** — `python -m agent.cli.<name> …` — because every path a
> script uses is derived from the selected profile under `runs/<profile>/`, not from the working
> directory. Recording, deployment and GAIL additionally assume Windows.

## 1. Choosing a profile

| | `roblox` | `hajime_ippo` |
|---|---|---|
| `process_name` | `RobloxPlayerBeta` | `rpcs3` |
| `exe_path` / `rom_path` | `None` / `None` in the profile (window must already exist) | `None` / `None` in the profile; the emulator and ISO come from `agent/config/local.py` |
| `actions.num_actions` | 9 | 18 |
| `actions.input_mode` | `keyboard_mouse` | `gamepad` (declared explicitly) |
| inference cap | `deploy.fps = 60` (enforced) | `deploy.fps = 30` (enforced) |
| action-width policy | `actions.width_policy = "strict"`, enforced by the shared loader `agent/utils/demos.py` | also `"strict"` — which is why the two 7-wide files in the retained corpus are refused (§12, item 2) |
| Recorded benchmark evidence | none | **all of it** (`runs/hajime_ippo/models/comparison_results.md`, the `Model_Comparison` experiment, the ≈293-epoch log) — even though the four files that produced it were recorded under the generic configuration |

This section used to diff the two near-duplicate packages: 10 of the 18 shared modules were
byte-identical and 8 differed, in the `keyboard_mouse` emission branch, the action-width reconciliation,
the `aggressiveness` branch and (for one script) the experiment name alone. That fork has been collapsed
into `agent/`, so there is one recorder, one trainer selected by `--arch`, one deployment path and one
demonstration loader; the remaining difference between the two titles is their profile — input mode,
action width and mapping table — and emission is driven by that table on both.
**Work with `--profile hajime_ippo` to reproduce or extend the published comparison**: every retained
checkpoint, log and MLflow run lives under `runs/hajime_ippo/`.

## 2. Editing `agent/config/profiles/hajime_ippo.py`

Field-by-field, with what actually consumes it. "Read by nothing" means the field is decorative: editing
it changes no behaviour.

| Field | Read by | Notes |
|---|---|---|
| `GAME_CONFIG.process_name` | `game_env.py:48`, matched at 230-249 | case-insensitive **substring** of the owning process name; first matching window in enumeration order wins |
| `GAME_CONFIG.exe_path` | `game_env.py:87-92` | only used if no window was found; then `Popen([exe_path, rom_path])`. `None` in both profiles; supply it in `agent/config/local.py` |
| `GAME_CONFIG.rom_path` | `game_env.py:89-90` | passed as the emulator's first argument; also a `local.py` override key |
| ~~`capture.width` / `.height`~~ | **removed** | these were dead fields (assigned and never used); they are no longer in either profile, and the grab region comes from the real window rect |
| `capture.internal_width` / `.internal_height` | `game_env.py:53-54` and `_get_observation` (152-173) | the resize target and the observation shape. Changing them changes `flatten_dim` and breaks the Impala-CNN head unless it is divisible by 8 |
| `capture.target_fps` | `game_env.py:55`, `camera.start(target_fps=…)` | **now honoured twice**: it also sets the frame budget `step()` sleeps out (198-203) and the recorder's own pacing (`agent/cli/record.py:210-213`) |
| `capture.buffer_len` | `game_env.py:56`, `dxcam.create(max_buffer_len=…)` | `1` = lowest latency |
| `window_offset.left/top/right/bottom` | `_window_region`, `game_env.py:109-116` | compensates title bar and emulator borders. Defaults `20, 100, 0, 0` |
| `actions.num_actions` | `game_env.py:60`, `agent.config.num_actions`, `HumanInput`, `load_demos` | defines the `MultiBinary` width **and** the width every demonstration file must have under `strict`; `agent.config.num_actions` (87-96) refuses a profile whose mapping count disagrees |
| `actions.input_mode` | `game_env.py:61`, `emission.build_emitter` (169-182) | `gamepad` or `keyboard_mouse`. Both profiles declare it explicitly; there is no silent default to rely on |
| `actions.mappings` | `agent/utils/emission.py` (both input modes) and `agent/utils/input_map.py` | **drives emission in gamepad mode as well as recording**, so the names in the Hajime table are live rather than inert. Schema: `kind` / `button` / `stick` / `axis` / `value` / `trigger`, validated by `emission.validate_mappings`; conflicting flags on one axis sum and clamp |
| `GAME_CONFIG.recording.max_trajectories` | `agent/cli/record.py:139`, overridable by `--max-trajectories` | the old `TRAINING_CONFIG.max_trajectories` was read by nothing; the value moved here and the recorder obeys it |
| `TRAINING_CONFIG.batch_size` / `.epochs` / `.learning_rate` | argparse *defaults* only (`agent/cli/train.py:181-183`) | overridden by `--batch` / `--epochs` / `--lr` |
| `TRAINING_CONFIG.window_size` | `agent/utils/architectures.py:110-112` → `policy_kwargs_for` | feeds `TemporalAttentionLSTM(window_size=…)`; it was unreachable while the value was hard-coded in `game_env.py` |
| `TRAINING_CONFIG.dagger_iterations` | `agent/cli/dagger.py:140-142` | the default for `--rounds`; the DAgger path is no longer a stub (§9) |
| ~~`TRAINING_CONFIG.demo_path` / `.model_path` / `.train_path`~~ | **removed** | dead as fields; the locations are derived from the profile by `agent/utils/paths.py` (`runs/<profile>/{demos,models,logs,mlruns}`), overridable with `IMITATION_RUNS` / `--runs-root` |
| `GAME_CONFIG.input.*` | `agent/utils/input_map.py:113-141, 180-188` | `INPUT_CONFIG` was flattened into `GAME_CONFIG["input"]`: `deadzone`, `camera_half_deadzone`, `camera_full_deadzone`, `trigger_threshold`, `keyboard_keys`. Its two dead fields (`camera_sensitivity`, `input_delay`) are gone |
| `GAME_CONFIG.deploy.*` | `agent/cli/deploy.py:59-61, 89-93` | `fps` is enforced by the loop, `aggressiveness` is read from this dictionary (the *top-level* lookup that made the branch dead is gone), and `attack_buttons` names the mapping indices it multiplies |

The minimum viable edit set is `process_name`, `window_offset`, `capture.internal_*` (leave at 128
unless you also touch the encoders), `actions.num_actions`, `actions.input_mode`, `actions.mappings` —
plus `exe_path`/`rom_path` in `agent/config/local.py` if the title should be launched for you.

## 3. Recording demonstrations

### 3.1 Protocol

Start the game or emulator **first** (otherwise the environment waits 120 s and then raises
`WindowNotFoundError` — [SETUP.md §7](SETUP.md#7-game-and-emulator-prerequisites)), then
`python -m agent.cli.record --profile hajime_ippo`. A 300×340 pygame window titled
`Imitation Player - Capture` appears with the upscaled 128×128 capture, an FPS readout,
`Demos: n/10`, a `RECORDING`/`IDLE` status, the live action-bit readout and the prompt
`[K] toggle record   [ESC] save & exit`. Key bindings, exactly as implemented:

| Input | Effect | Line |
|---|---|---|
| `K` | toggle recording on/off (0.3 s debounce) | `agent/cli/record.py:177-182` |
| `ESC`, window close button (`pygame.QUIT`) or Ctrl+C | exit; the in-progress buffer is flushed first | `record.py:169-175, 214-217` |
| `↑ ↓ ← →` / `i`, `o`, `p` (+ `u`, `j`, `k`, `;`) | keyboard stand-ins for the named mappings — `UP`/`DOWN`/`LEFT`/`RIGHT` and `CROSS`/`CIRCLE`/`SQUARE`, `TRIANGLE`, `L2`, `R2`, `L3`, `R3` — used only when no physical pad answers XInput | `agent/utils/input_map.py:52-58, 206-211` |
| physical pad on XInput slot 0 | every mapping kind gets a source: D-pad masks `0x0001/2/4/8` or a stick past `input.deadzone` → bits 0-3; `0x1000/0x2000/0x4000` (A/B/X) → bits 4-6; `bLeftTrigger`/`bRightTrigger` past `trigger_threshold` → bits 7-8; `0x0040` (L3) → bit 9; `sThumbRX/RY` past `camera_half_deadzone`/`camera_full_deadzone` → bits 10-17 | `input_map.py:113-141` |
| configured keys / mouse buttons | in `keyboard_mouse` mode, `keyboard.is_pressed(mapping["key"])` and `mouse.is_pressed(mapping["button"])` per mapping index | `input_map.py:214-228` |
| `input.keyboard_keys` | overrides any entry of the stand-in table by mapping **name**, so an unmapped bit (a camera axis, a trigger) can be given a key | `input_map.py:180-188` |

**`ESC` does save.** `TrajectoryRecorder.finish()` runs in `main()`'s `finally` block
(`record.py:102-106, 216-217`), so `ESC`, the window close button and Ctrl+C each flush an in-progress
recording to a file — previously only the `K` stop-transition wrote one, and the overlay promised what the
code did not do. The loop additionally paces itself to `capture.target_fps`, and `stop_and_save()` refuses
to write a buffer whose observation and action counts have desynchronised (`record.py:77-81`).

### 3.2 What the recorder writes

`runs/<profile>/demos/demo_<n>_<YYYYMMDD>_<HHMMSS>.pt` — `<n>` is the zero-based index of the saved
trajectory, the timestamp is `time.strftime("%Y%m%d_%H%M%S")` at the moment of the stop. The payload is
`th.save([traj])`: a **list holding one `imitation.data.types.Trajectory`** dataclass with fields `obs`,
`acts`, `infos` (`None`) and `terminal` (`False`), so loading one requires the `imitation` package. Shapes
measured directly from the five corpus files:

| File | `obs` | `acts` |
|---|---|---|
| `demo_0_20260523_221206.pt` | `(6302, 4, 128, 128)` uint8 | `(6301, 18)` float32 |
| `demo_1_20260523_221830.pt` | `(7069, 4, 128, 128)` uint8 | `(7068, 18)` float32 |
| `demo_0_20260523_224241.pt` | `(8150, 4, 128, 128)` uint8 | `(8149, 7)` float32 |
| `demo_1_20260523_224751.pt` | `(5644, 4, 128, 128)` uint8 | `(5643, 7)` float32 |
| `demo_0_20260619_103226.pt` (hajime) | `(6151, 4, 128, 128)` uint8 | `(6150, 18)` float32 |

`obs` has **one row more than `acts`** by design: the buffer is seeded with the frame that `reset()`
returns (`record.py:59-65`) and every step appends the frame its action produced, so the final observation
is the outcome of the final action — the files above were written by a recorder that duplicated the first
frame instead, which is where their extra row comes from. Stacking is 4-deep, done *at capture time*.

### 3.3 Measured corpus, and what it implies

Generic corpus: **27 161 action frames** across four files, ≈ 7.5 min of play at 60 fps. Hajime:
**6 150 frames in one file**. Per-bit marginal firing rates (mean of `acts`):

| Bit | Semantic | Generic files | Hajime |
|---|---|---|---|
| 0 / 1 | up / down | 0.428 – 0.527 / 0.308 – 0.410 | 0.266 / 0.087 |
| 2 / 3 | left / right | 0.077 – 0.101 / 0.182 – 0.220 | 0.195 / 0.065 |
| 4 / 5 / 6 | A-Cross / B-Circle / X-Square | 0.054 – 0.266 / 0.005 – 0.008 / 0.002 – 0.004 | 0.032 / 0.069 / 0.020 |
| **7 – 17** | triggers, stick press, camera | **exactly 0.0 in every file** | **exactly 0.0** |

Distinct joint action vectors per file: 35, 36, 37, 39, and **24** in the hajime session — at ~27 k frames
the corpus is two orders of magnitude larger than its own action diversity. Two structural problems
follow, and README §6.2 shows they dominate every reported loss.

**The dead bits 7-17 were unreachable by code — they no longer are.** The old gamepad branch wrote only
indices 0-6, so no amount of playing set bit 7 (LT), 8 (RT), 9 (L3) or 10-17 (right stick). That gap is
closed in `agent/utils/input_map.py`: triggers are read from `bLeftTrigger`/`bRightTrigger` against
`input.trigger_threshold`, the stick press from the `0x0040`/`0x0080` masks, and the camera axes from
`sThumbRX`/`sThumbRY` against `camera_half_deadzone`/`camera_full_deadzone`, all resolved through the
profile's mapping table rather than hard-coded indices. The bits are still exactly zero in every existing
file, because those files predate the change — closing them is now a re-recording task, not a code change.
`keyboard_mouse` with 9 mappings remains the other option, where every bit is also reachable.

**Class imbalance is fixable by recording behaviour:**

1. **Many short clips, not one long session.** `K`, play one specific situation for 60-90 s, `K`. Each
   stop writes a separate file, so the shuffler sees independent trajectories rather than one
   autocorrelated block; the loop caps at 10 files per process, so run several sessions.
2. **Deliberately actuate the rare bits** — 5 and 6 fire below 1 %. A defence-only clip, a whiffed-combo
   clip, a cornered clip and an approach clip beat five more neutral clips.
3. **Cover states, not frames:** more frames of the same kind teach nothing new; the missing *action
   combinations* do.
4. **Keep widths consistent** — do not reconfigure `num_actions` between recordings into the same
   `runs/<profile>/demos/` folder; two of the four generic files are 7-wide because of exactly that, and
   the default `strict` policy now refuses to train on them (§12).
5. Verify with `python -m agent.cli.train --profile hajime_ippo --arch naturecnn --epochs 1 --batch 32
   --device cpu --width-policy coerce`, whose `per-bit firing rate:` block prints the marginal rate for
   every named mapping and flags the bits that never fire.

## 4. Training

### 4.1 `agent/cli/train.py` — the primary BC entry point

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--epochs` | int | `TRAINING_CONFIG["epochs"]` = 100 | BC epochs, passed to `bc_trainer.train(n_epochs=…)` |
| `--batch` | int | `TRAINING_CONFIG["batch_size"]` = 384 | demonstration minibatch size |
| `--lr` | float | `TRAINING_CONFIG["learning_rate"]` = 1e-4 | constant schedule (`optimizer_kwargs={"lr": …}`) |
| `--arch` | choice | `naturecnn` | feature extractor from the registry in `agent/utils/architectures.py` (`naturecnn`, `lstm`, `transformer`, `impoola`, `impala`, `resnet18`) |
| `--device` | str | `None` → `cuda` when available, else `cpu` | torch device; an explicit value is honoured as given |
| `--model_path` | str | `None` | warm start from an existing `.zip` (§7) |

Backbone is `ActorCriticCnnPolicy` (NatureCNN) unless `--arch` supplies an extractor. Writes
`runs/<profile>/models/<prefix>.zip` and `runs/<profile>/logs/progress.csv`; logs to experiment
`f"{profile}_imitation_bc"` (e.g. `hajime_ippo_imitation_bc`) under a run named after `--arch`. The
retained runs in `runs/hajime_ippo/mlruns/` are the ones that all reported into
`Hajime_no_Ippo_Imitation_Learning`, whichever profile had produced the corpus.
`--width-policy` and `--runs-root` override the profile's demonstration policy and the runs root.

### 4.2 Architectures and entry points

| Invocation | Extractor | Flags | Experiment / run name | Checkpoint written |
|---|---|---|---|---|
| `python -m agent.cli.train` (default `--arch naturecnn`) | NatureCNN | `--epochs --batch --lr --arch --device --model_path --width-policy` | `hajime_ippo_imitation_bc` / `naturecnn` | `bc_policy.zip` |
| `--arch lstm` | `TemporalAttentionLSTM` | as above | `hajime_ippo_imitation_bc` / `lstm` | `bc_policy_lstm.zip` |
| `--arch transformer` | `VisionTransformerExtractor` | as above | `hajime_ippo_imitation_bc` / `transformer` | `bc_policy_transformer.zip` |
| `--arch impoola` | `ImpoolaCNNExtractor` | as above (same `TRAINING_CONFIG` defaults as every other architecture) | `hajime_ippo_imitation_bc` / `impoola` | `ImpoolaCNN_policy.zip` |
| `--arch impala` / `--arch resnet18` | `ImpalaCNNExtractor` / `ResNet18Extractor` | as above | `hajime_ippo_imitation_bc` / `impala`, `resnet18` | `Impala_CNN_policy.zip` / `ResNet18_policy.zip` |
| `python -m agent.cli.benchmark` | all six | `--epochs --batch --lr --device --archs` (defaults 10/384/1e-4) | reads and writes the store; opens no run of its own (§11) | `runs/<profile>/models/<Name>_policy.zip` |

The four per-architecture scripts and the `train_imiation.py` minimal loop are gone, so the experiment
names they used (`Generic_Agent_LSTM_IL`, `Generic_Agent_Transformer_IL`, `BC_Training_Simple`) appear
nowhere in `runs/hajime_ippo/mlruns/`: neither recurrent/transformer architecture was ever trained through
a standalone script here, only through the benchmark harness. Checkpoint prefixes are unchanged, which is
what keeps the retained `Model_Comparison` runs readable by `agent.cli.deploy`.

### 4.3 Budgets, and the evidence for them

```powershell
# Architecture study (regenerates README §10.1 and runs/hajime_ippo/models/comparison_results.md)
python -m agent.cli.benchmark --profile hajime_ippo --epochs 10 --batch 384 --lr 1e-4
# A converged single model
python -m agent.cli.train --profile hajime_ippo --arch naturecnn --epochs 300 --batch 384 --lr 1e-4 --device cuda
```

10 epochs is a **screening budget, not a training budget**. The benchmark stops every encoder between
2.90 and 3.48, while the one retained long run on the same corpus (`runs/hajime_ippo/logs/progress.csv`,
42 logged rows spanning `bc/epoch` 0 → 292) falls from `bc/loss` 4.851 to **1.330** and lifts
`bc/prob_true_act` from 0.0078 to **0.483**. Ten epochs is ≈ 3 % of what this corpus can absorb and the
loss is still falling steeply at epoch 10, so a 10-epoch figure does not describe a converged model, and
encoders separated by <0.13 there are not separated at all. Always report the budget with the loss (§13).

## 5. Reading the metrics

All `bc/*` keys are emitted by `imitation`'s own loss calculator
(`imitation/algorithms/bc.py:126-155`) into the SB3 logger and forwarded to MLflow by
`tracking.MLflowOutputFormat` (`agent/utils/tracking.py:45-62`, installed by `agent/cli/train.py:92-99`).
`BC.train` dumps every `log_interval` batches (default
500, `bc.py:388`), which is why the retained `progress.csv` rows sit at `bc/batch` 0, 500, 1000, ….

| Metric | Definition in code | Informative? |
|---|---|---|
| `bc/neglogp` | `-log_prob.mean()`, where `log_prob` is `Bernoulli.log_prob(acts).sum(dim=1)` → **mean over the batch of the sum over bits**, in nats | yes — the actual BC objective |
| `bc/prob_true_act` | `exp(log_prob).mean()` → mean joint probability assigned to the demonstrated action vector | **the most interpretable fit metric**; 1/2ⁿ for a uniform predictor, 0.48 at epoch 292 |
| `bc/loss` | `neglogp + ent_loss + l2_loss` | yes, but it is a composite; watch it drift from `neglogp` only if entropy changes fast |
| `bc/entropy` | `Bernoulli.entropy().sum(dim=1).mean()` → summed over bits. ≈ 12.5 nats for 18 unbiased bits; observed 4.85 → 1.43 | yes — a healthy collapse. If it reaches ~0 early the policy has collapsed to a marginal predictor |
| `bc/ent_loss` | `-1e-3 × bc/entropy` (`ent_weight` default `1e-3`, `bc.py:286`) | no, derivative |
| `bc/l2_norm` | `½ Σ w²` over all policy parameters | mildly — a rising curve with flat loss suggests divergence |
| `bc/l2_loss` | `l2_weight × l2_norm` with `l2_weight` default **`0.0`** (`bc.py:287`). Verified identically `0.0` in all 42 logged rows; new runs omit it (`tracking.DEFAULT_SKIP_KEYS`) | **dead** |
| `bc/epoch`, `bc/batch`, `bc/samples_so_far` | loop counters written by `bc.py:231-234`; `samples_so_far` is the cumulative samples consumed (`bc.py:163-167`), so its final logged value 7 872 384 is exactly `20 501 × 384` | bookkeeping |
| `batch_size` | the logger's own key, equal to `--batch` (`bc.py:231`) | bookkeeping |
| `final_loss` | the **last value** of `bc/loss` seen by `MetricCapture` (`agent/utils/tracking.py:30-41`), i.e. one minibatch, not an epoch average | yes, but noisy by construction |
| `training_time_s` | wall seconds around `bc_trainer.train()` | yes |
| `num_params` | `sum(p.numel() for p in policy.parameters() if p.requires_grad)` | yes |
| `model_size_mb` | `os.path.getsize(zip) / 1024²` | yes |

**No accuracy metric is logged anywhere.** Earlier documentation claimed MLflow recorded per-action
accuracy; grep over all `bc/*` keys the scripts write, and over the retained `progress.csv` header, finds
none. `bc/prob_true_act` is a joint probability, not an accuracy, and is bounded above by the marginal
distribution of the corpus.

A healthy curve: `bc/loss` falls steeply through the first epoch and is still falling at epoch 10;
`bc/entropy` decays steadily from ≈ 12.5 nats (18 bits) or ≈ 6.2 (9 bits) toward 1-2; `bc/prob_true_act`
rises monotonically from ~`1/2ⁿ` toward 0.3-0.5; `bc/neglogp` exceeds `bc/loss` by exactly
`1e-3 × bc/entropy`. If `prob_true_act` stalls below ~0.08 while entropy collapses, the policy has
memorised the marginal, not the conditional — the README §6.2 imbalance signature, not an architecture
property.

**Comparability warning.** Because the objective is *summed over bits*, absolute losses are only
comparable between runs sharing `num_actions`. The published benchmark ran at **9** (the generic config);
the Hajime config declares **18**. A 9-bit loss near 2.9 and an 18-bit loss near 2.9 are different
achievements, and neither is comparable to a mean-per-bit cross-entropy.

## 6. MLflow UI

```powershell
mlflow ui --backend-store-uri file:runs/hajime_ippo/mlruns
```

Open <http://localhost:5000>, from the repository root. The scripts set their own tracking URI from the
profile (`agent/utils/tracking.configure_store`), so there is **one store per profile** instead of one per
package, and no CWD dependence. The `artifact_location` recorded in the retained `mlruns/*/meta.yaml`
files still points at the per-package paths they were written under, so historical runs report their old
absolute locations while resolving through `runs/hajime_ippo/mlruns/`. Experiments present on disk:

| Experiment | Former store | Runs |
|---|---|---|
| `Model_Comparison` | generic | 26 |
| `Hajime_no_Ippo_Imitation_Learning` | generic | 5 |
| `Hajime_no_Ippo_Imitation_Learning` | hajime | 1 |
| `Default` | hajime | 0 |

Filter per-batch curves by prefix: the logger namespaces them `<Model>/bc/…` using the registry name
(`agent/utils/tracking.py:48-57`), e.g. `Impoola_CNN/bc/loss` against `ResNet18/bc/loss`. The store is
**git-ignored** (`.gitignore`: `mlruns/`, `runs/*/mlruns/`, `runs/*/demos/`), so these runs exist only on
the workstation that produced them.

## 7. Transfer / warm start

```powershell
python -m agent.cli.train --arch naturecnn --epochs 50 --batch 384 --lr 1e-4 --model_path runs/hajime_ippo/models/bc_policy.zip
```

`agent/cli/train.py:53-58`: if the path exists, `ActorCriticCnnPolicy.load(model_path, device=…)` is
called and the instance is handed to `BC(policy=…)`. Three consequences. (1) The **architecture is
inherited from the checkpoint**: `--arch` is not consulted once `--model_path` resolves, so an inconsistent
pair silently trains the checkpoint's encoder. (2) Optimiser
state is not restored and `lr_schedule` is rebuilt, so the rate restarts at `--lr`. (3) The **output file is
chosen by `--arch`, not by the input** — `--arch naturecnn` writes `bc_policy.zip` and therefore overwrites
its own seed, while `--arch impoola` writes `ImpoolaCNN_policy.zip` beside it; copy the seed first if you
want to keep it. A missing path **aborts** with `[!] --model_path <path> does not exist` rather than
carrying on from scratch. `--model_path` is available for every architecture through this one script;
`agent/cli/benchmark.py` does not expose it.

## 8. Deploying a policy

```powershell
python -m agent.cli.deploy --arch naturecnn      # runs/hajime_ippo/models/bc_policy.zip
python -m agent.cli.deploy --arch lstm           # bc_policy_lstm.zip
python -m agent.cli.deploy --arch transformer    # bc_policy_transformer.zip
python -m agent.cli.deploy --arch impoola        # ImpoolaCNN_policy.zip
```

| `--arch` | Model it reads | Keys | Enforced `deploy.fps` |
|---|---|---|---|
| `naturecnn` | `bc_policy.zip`, else the highest numbered `bc_policy<N>.zip` via `resolve_checkpoint` | `K` toggle AI/manual, `ESC` quit, manual play through `HumanInput` (physical pad, or the keyboard stand-ins when no pad answers) | roblox 60 / hajime_ippo 30 |
| `lstm` | `bc_policy_lstm.zip`, else the highest numbered `bc_policy_lstm<N>.zip` | as above | as above |
| `transformer` | `bc_policy_transformer.zip`, else its numbered fallback; `impoola`, `impala` and `resnet18` resolve to `ImpoolaCNN_policy.zip`, `Impala_CNN_policy.zip` and `ResNet18_policy.zip` the same way | as above | as above |

Verified facts about the deploy path:

- **`deploy.fps` is enforced.** The loop times each step and sleeps the remainder of `1 / fps`
  (`agent/cli/deploy.py:98, 124-126`), so the declared cap is behavioural: 30 in `hajime_ippo`, 60 in
  `roblox`. The old `MAX_FPS` script constants — declared in three files, never read, leaving the loop
  unpaced — are gone.
- **The `aggressiveness` branch is reachable.** `agent/cli/deploy.py:61` reads
  `GAME_CONFIG["deploy"]["aggressiveness"]`, the dictionary the profile actually defines it in; the
  former top-level `GAME_CONFIG.get("aggressiveness", 1.0)` lookup against a value filed in `INPUT_CONFIG`
  always missed and made the path dead. Both profiles ship `1.0`, which disables sharpening by
  configuration, so set e.g. `2.0` to exercise it. When it fires, the work is done by
  `PolicyRunner.sharpened` (`agent/utils/utils.py:28-46`): `sigmoid(action_net(...))`, multiply the
  probabilities of the indices named in `deploy.attack_buttons`, clamp to 1.0, sample
  `(rand(len(probs)) < probs)`. That is still not logit sharpening and still bypasses SB3's action
  post-processing ([ARCHITECTURE.md §5](ARCHITECTURE.md#5-inference-path-agentclideploypy)).
- **The benchmarked encoders are deployable.** The registry's `checkpoint_prefix` is what `--arch`
  resolves, so `--arch impoola` reads `runs/<profile>/models/ImpoolaCNN_policy.zip` directly; the
  copy-onto-`bc_policy.zip` workaround the old `run_ai*.py` scripts required is obsolete, and
  `--model <path>` bypasses resolution altogether.
- Recurrent models are detected by the presence of `reset_hidden`
  (`agent/utils/utils.py:17`) and routed through `PolicyRunner` (which replaced `LSTMWrapper`); its
  `reset()` fires on the `K` toggle, and since `done` is never `True`, nowhere else.

## 9. DAgger (interactive correction)

```powershell
python -m agent.cli.dagger --arch naturecnn --rounds 3 --seconds 120
```

Bindings (`agent/cli/dagger.py:68-80`; the overlay reads
`HUMAN (recording)   118.3s left` / `[L] take over   [ESC] stop`):

| Key | Effect |
|---|---|
| `L` (held) | takes authority from the policy; the frames you steer are buffered. Releasing it flushes that segment to a file and resets the recurrent state |
| `ESC` | stop collecting. The in-progress buffer **is** flushed — the `finally` block calls the same `_flush` the release path uses (`dagger.py:93-95`) |
| window close button | exits (`pygame.QUIT`), with the same flush |

Worked example: start a round and let the policy play; the moment it errs hold `L` to take over, play the
recovery, release `L` to hand back — that segment is written immediately. Repeat until the round's
`--seconds` budget expires, then the script retrains. Output is
`runs/<profile>/demos/demo_dagger_<profile>_<YYYYMMDD>_<HHMMSS>.pt` — one `Trajectory` in a list, format as
§3.2, with the recorder's `N+1` observation convention reproduced (`dagger.py:105-108`) — and
**only frames recorded while `L` was held are stored**; the policy's own actions are not logged at all,
which is the correct DAgger label but means a round with no corrections writes no file. The round is
bounded by the `--seconds` time budget rather than by the old `MAX_TRAJ = 10` file cap.

The `demo_dagger_*` prefix matches the `demo*.pt` glob used by every loader, so the next training run
picks the corrections up automatically — which the script now does itself. The human correction vector is
read through the same `agent.utils.input_map.HumanInput` as the recorder (`dagger.py:160`), so a physical
pad works in gamepad mode and the keyboard stand-ins cover the case where no pad answers; the old split in
which `run_dagger.py` polled only the keyboard while the recorder polled XInput is gone.

**Honest status of the loop.** The aggregation-and-retrain half now exists. The two `pass` stubs the old
scripts advertised are gone with the scripts that held them: `--dagger` no longer appears anywhere in the
CLI, and `agent/cli/dagger.py` calls `agent.cli.train.main` after each collection round
(`dagger.py:170-182`), printing the aggregated corpus size and its marginal baseline first, unless
`--collect-only` is passed. DAgger is therefore a **one-command loop** for `--rounds`
(default: `TRAINING_CONFIG["dagger_iterations"]` = 3) iterations. No DAgger iteration has ever been
executed here — the store contains no run attributable to it.

## 10. GAIL (adversarial imitation)

```powershell
python -m agent.cli.train_gail --profile hajime_ippo --timesteps 100000
```

| Flag | Default | Effect |
|---|---|---|
| `--timesteps` | 100 000 | total *environment* steps for the PPO generator |
| `--model_path` | `None` | `PPO.load(model_path, env=…)` as the generator seed |

Fixed defaults, now exposed as flags (`--demo-batch-size`, `--gen-batch-capacity`,
`--disc-updates-per-round`; `agent/cli/train_gail.py:50-54`): PPO generator with `batch_size=64`,
`learning_rate=3e-4`, `n_steps=1024`, `ent_coef=0.01`, `gamma=0.99`, `ActorCriticCnnPolicy`;
`BasicRewardNet` discriminator with `RunningNorm`; `demo_batch_size=64`,
`gen_replay_buffer_capacity=2048`, `n_disc_updates_per_round=4`. Logs to
`runs/<profile>/logs/gail/progress.csv` and to experiment `f"{profile}_gail"` (e.g. `hajime_ippo_gail`) /
run `gail`; saves `runs/<profile>/models/gail_policy.zip` (the *policy*, not the full PPO).

**Warnings.** The CLI builds its environment with `build_config(config, dummy=False)`
(`agent/cli/train_gail.py:71`) because the generator must interact with the real game: a dummy environment
now `reset()`s (it returns a zero frame) but refuses to `step()` with an explicit `RuntimeError`
(`agent/utils/game_env.py:189-193`). **GAIL therefore requires the game running, the window found and the
capture path working: it is impossible headless and impossible without ViGEmBus.** The 2048-step generator
buffer against 100 000 requested timesteps keeps ~2 % of the trajectory in the replay window, and the
reward is discriminator-derived on a 27 k-frame corpus. Finally: **GAIL has never completed a run in this
project and never started one on this machine** — no `GAIL_Run` and no `gail` log directory exists in the
retained store (verified over `runs/hajime_ippo/mlruns`). Read this whole section as a description of
code, not a result.

## 11. Benchmark

```powershell
python -m agent.cli.benchmark --profile hajime_ippo --epochs 10 --batch 384 --lr 1e-4  # full six-model study
python -m agent.cli.benchmark --archs impala resnet18   # two models trained, four read back from the store
```

A full run trains all six encoders in sequence, saves `runs/<profile>/models/<Name>_policy.zip` per
architecture, then calls `regenerate_readme()` and `write_report()`, which **rewrite two files**: (1) the
repository-root `README.md` — the table between `<!-- BENCHMARK_START -->` and `<!-- BENCHMARK_END -->` is
replaced and the line after the `**Conclusion**:` label regenerated from the measured values; nothing else
is touched, so keep both markers intact; (2) `runs/<profile>/models/comparison_results.md`, regenerated
wholesale including the *Experiment Configuration* block recording the epoch/batch/lr/device, the corpus
frame and trajectory counts, the action width and the distinct joint-action count actually used. Note that
the harness calls `agent.cli.train.train()` directly and opens **no MLflow run of its own**, so a
benchmark pass writes the report and the checkpoints but leaves the tracking store to `python -m
agent.cli.train`; `--archs` columns are filled from whatever the store already holds.

**`--archs` behaviour.** It trains only the named architectures; the remaining columns are read back from
the MLflow store by `historical_from_store` (`agent/cli/benchmark.py:58-88`), which takes the most recent
completed run of that architecture and reports how many runs it saw. An architecture with no run in the
store renders as `-` with a `Runs available` count of 0 — it is never substituted from a constant. The old
`--only-new` mode did exactly that: it trained `Impala_CNN` and `ResNet18` only and printed hard-coded
numbers for the other four without consulting the store, mixing fresh measurements with figures frozen from
an earlier session; and because both packages carried the same dict, running it in one package printed the
other's corpus numbers. A related trap went with it: `train_agent_impoola.py` embedded the NatureCNN,
CNN_LSTM and ViT baselines in its own report writer and rewrote `comparison_results.md` as a four-column
table whenever it ran. Neither script nor dict survives the collapse into `agent/cli/benchmark.py`.

## 12. Known broken things

| # | Defect | Where |
|---|---|---|
| 1 | **Fixed.** The minimal BC loop is gone: no misspelled filename, and no `demos*.pt` glob that never matched the recorder's `demo*.pt`. Its `mlflow.log_artifact('./models/bc_policy.zip')` tail went with it | folded into `agent/cli/train.py`; the single pattern is `DEMO_GLOB = "demo*.pt"` in `agent/utils/demos.py:20` |
| 2 | **Fixed.** Mixed demonstration widths are no longer reconciled silently. The default `strict` policy raises `DemoError` naming the file and both widths; `coerce` truncates/pads but prints a warning per file. The published benchmark still ran on a 9-bit label space assembled from Hajime recordings | `agent/utils/demos.py:37-99`, `actions.width_policy`, `--width-policy` |
| 3 | **Fixed.** A demo file that raises while loading is collected and reported as one `DemoError` listing every failure, instead of `except Exception` + `continue` | `agent/utils/demos.py:60-96` |
| 4 | **Partly fixed.** A failed capture still returns the previous frame (black if none), but it now increments `self.dropped_frames`; the counter is neither printed nor carried into the demonstrations, so a stored stale frame remains indistinguishable from a legitimately black screen | `agent/utils/game_env.py:158-165` |
| 5 | **Fixed in code, open in data.** Every gamepad mapping now has a physical source (triggers, stick press, right-stick camera axes), so bits 7-17 are recordable; they are still identically zero in the corpus because the existing files predate the change | `agent/utils/input_map.py:113-141` |
| 6 | **Fixed.** `ESC`, the window close button and Ctrl+C all flush the in-progress trajectory through `TrajectoryRecorder.finish()`, so the overlay's `[ESC] Save & Exit` is now accurate; the DAgger loop flushes in its own `finally` | `agent/cli/record.py:102-106, 216-217`; `agent/cli/dagger.py:93-95` |
| 7 | **Fixed.** `aggressiveness` is read from `GAME_CONFIG["deploy"]`, the dictionary that defines it, so the inference-time sharpening branch executes when configured. Both profiles currently ship `1.0` | `agent/cli/deploy.py:61`, `agent/utils/utils.py:28-46` |
| 8 | **Fixed.** The deployment loop sleeps the remainder of the `deploy.fps` budget, so the declared cap is enforced; the unread `MAX_FPS` constants and unused `loop_start` are gone. The recorder paces itself the same way | `agent/cli/deploy.py:98, 124-126`; `agent/cli/record.py:210-213` |
| 9 | **Replaced by an explicit refusal.** A dummy environment now `reset()`s (a zero frame) but raises `RuntimeError` on `step()`, saying it cannot actuate a game; the trainers never step it, and GAIL asks for a live one on purpose | `agent/utils/game_env.py:189-193`, `148-152` |
| 10 | **Fixed.** `step()` no longer indexes `actions[10:18]`: emission walks the mapping table, and the vector width is validated against `num_actions` up front, so a narrow profile raises `ValueError` with a clear message instead of `IndexError` mid-emission | `agent/utils/game_env.py:177-196`, `agent/utils/emission.py` |
| 11 | **Fixed by removal.** The stack is built once, in `agent/cli/common.wrapped_env`, so every entry point trains and deploys against `(4, 128, 128)` observations matching the demonstrations | `agent/cli/common.py:68-74` |
| 12 | **Fixed.** There is one loader for both profiles, so neither can survive a mixed-width corpus by accident; the shipped files fail under `hajime_ippo`'s `strict` policy until `--width-policy coerce` is given | `agent/utils/demos.py` |

**Correction to a widely-repeated claim.** README §12.6 and ARCHITECTURE.md §5 used to say `get_last_index`
"compares checkpoint suffixes lexicographically, so `bc_policy9.zip` outranks `bc_policy10.zip`". That was
**not what the code did**: the implementation extracted the digits and folded
`max(last_index, int(m.group(1)))` — a numeric maximum, so no such inversion occurred. The real defect was
different, and is now gone with it: the directory that deployment asked it to scan
(`models/steps/`, from `STEPS_PATH = MODEL_PATH + "steps"`) was created by no script, so the fallback always
returned `-1` and only the fixed-name `bc_policy.zip` branch ever resolved. `get_last_index` now lives in
`agent/utils/checkpoints.py:15-28`, sorts numerically there and in `list_checkpoints`, and
`resolve_checkpoint()` (46-55) prefers the final `<prefix>.zip` before falling back to the highest numbered
checkpoint — rooted at `runs/<profile>/models/` rather than a relative path.

## 13. Reproducibility checklist

Quote a number from this project only if you can state all of the following.

1. **Profile and invocation** — `--profile hajime_ippo` or `--profile roblox`, run as
   `python -m agent.cli.<name>` from the repository root, plus `--runs-root` (or `IMITATION_RUNS`) if the
   default `runs/` root was overridden.
2. **Commit and interpreter** — `git rev-parse HEAD`; `python -c "import sys,torch; print(sys.version,
   torch.__version__, torch.version.cuda)"`.
3. **Corpus** — the files matched by `demo*.pt`, their frame counts, their original widths and the
   `num_actions` they were reconciled to (item 2 of §12 makes the historical reconciliation non-obvious);
   keep the printed corpus summary — frame count, trajectory count, distinct joint actions, the marginal
   baseline and the per-bit firing rates — from a `--epochs 1` pass.
4. **Budget and device** — epochs, batch, lr, `cuda`/`cpu` and the GPU model. A 10-epoch loss is not a
   converged loss (§4.3), and `training_time_s` is meaningless without the hardware stated.
5. **Seed status** — only the demonstration shuffler is seeded (`np.random.default_rng(seed=42)`, recorded
   as the run parameter `seed`). No
   `torch.manual_seed`, `cudnn.deterministic` or `torch.use_deterministic_algorithms` call exists in the
   tree, so initialisation and CUDA kernel order are uncontrolled and the same command will not
   reproduce a loss to the third decimal.
6. **Metric identity and its limits** — the key name (`bc/loss` vs `final_loss` vs `bc/neglogp`), that
   it is summed over `num_actions` bits, that it is in-sample, and that no held-out split, accuracy,
   rollout or success rate exists behind it. The figure ranks optimisation ease and nothing else.
7. **Artefacts** — attach `runs/<profile>/models/comparison_results.md`, the MLflow run id and
   `runs/<profile>/logs/progress.csv`, noting that demonstrations, checkpoints and `runs/*/mlruns/` are all
   git-ignored (`*.pt`, `*.zip`) and therefore unobtainable from the repository; only
   `comparison_results.md` and the reference `progress.csv` are versioned.
