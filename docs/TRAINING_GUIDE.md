# Training Guide

> *The English successor to `GUIA_TREINAMENTO.md`, which the previous README referenced and which no
> longer exists in the tree.* Operator-facing and in pipeline order; rationale lives in
> [ARCHITECTURE.md](ARCHITECTURE.md), installation in [SETUP.md](SETUP.md), measurements in
> [RESULTS.md](RESULTS.md). All commands assume Windows, an active Python 3.11 virtual environment, and a
> **current working directory of `<pkg>/notebooks`** — every script inserts `".."` and `"../utils"` into
> `sys.path` and hard-codes `./demos/`, `./models/` and `file:../mlruns` relative to the CWD.

## 1. Choosing a package

| | `generic_agent` | `hajime_agent` |
|---|---|---|
| `process_name` | `RobloxPlayerBeta` | `rpcs3` |
| `exe_path` / `rom_path` | `None` / `None` (window must already exist) | absolute paths to one user's RPCS3 build and ISO |
| `actions.num_actions` | 9 | 18 |
| `actions.input_mode` | `keyboard_mouse` | absent → environment defaults to `gamepad` |
| `run_ai.py` inference cap | `MAX_FPS = 120` (declared) | `MAX_FPS = 30` (declared) |
| `train_agent.py` action-width reconciliation | present (`train_agent.py:155-177`) | **absent** (verified: no `expected_num_actions` in that file) |
| Recorded benchmark evidence | **all of it** (`models/comparison_results.md`, `mlruns/Model_Comparison`, the ≈293-epoch log) | none |

Measured by diffing all 18 shared paths: **10 are byte-identical** (`utils/utils.py`,
`utils/impoola_cnn.py`, `utils/new_architectures.py`, `utils/vision_transformer.py`,
`notebooks/compare_models.py`, `run_ai_lstm.py`, `run_ai_transformer.py`, `train_agent_impoola.py`,
`train_gail.py`, `train_imiation.py`) and **8 differ** (`config/game_config.py`, `utils/game_env.py`,
`record_trajectories.py`, `run_ai.py`, `run_dagger.py`, `train_agent.py`, `train_agent_lstm.py`,
`train_agent_transformer.py`). README §3.1 states nine identical files and names `train_agent_lstm.py`
among them; it differs in practice, in the experiment name alone (`Generic_Agent_LSTM_IL` vs
`Hajime_no_Ippo_LSTM_IL`). The substantive divergences are the `keyboard_mouse` emission branch (present
only in `generic_agent/utils/game_env.py`), the action-width reconciliation (only in
`generic_agent/notebooks/train_agent.py`) and the `aggressiveness` branch of `run_ai.py`.
**Work in `generic_agent` to reproduce or extend the published comparison**: the Hajime copy has never
been benchmarked, and fixes do not propagate automatically between the two copies.

## 2. Editing `config/game_config.py`

Field-by-field, with what actually consumes it. "Read by nothing" means the field is decorative: editing
it changes no behaviour.

| Field | Read by | Notes |
|---|---|---|
| `GAME_CONFIG.process_name` | `game_env.py:70`, matched at 322-337 | case-insensitive **substring** of the owning process name; first matching window in enumeration order wins |
| `GAME_CONFIG.exe_path` | `game_env.py:114-121` | only used if no window was found; then `Popen([exe_path, rom_path])` |
| `GAME_CONFIG.rom_path` | `game_env.py:116`, `train_imiation.py:180` | passed as the emulator's first argument |
| `GAME_CONFIG.capture.width` / `.height` | **read by nothing** | assigned to `self.width` / `self.height` at lines 76-77 and never used; the grab region comes from the real window rect |
| `capture.internal_width` / `.internal_height` | `game_env.py:78-79` and `_get_observation` | the resize target and the observation shape. Changing them changes `flatten_dim` and breaks the Impala-CNN head unless it is divisible by 8 |
| `capture.target_fps` | `game_env.py:80`, `camera.start(target_fps=…)` | also feeds `self.frame_time` (lines 107, 155), which is never read |
| `capture.buffer_len` | `game_env.py:81`, `dxcam.create(max_buffer_len=…)` | `1` = lowest latency |
| `window_offset.left/top/right/bottom` | `_get_window_region`, `game_env.py:157-165` | compensates title bar and emulator borders. Defaults `20, 100, 0, 0` |
| `actions.num_actions` | `game_env.py:87`, every `notebooks/*.py` that builds an action vector, and the loaders' width reconciliation | defines the `MultiBinary` width **and** the target width that demonstrations are truncated/padded to |
| `actions.input_mode` | `generic_agent/utils/game_env.py:91` + the recorder/deploy scripts | `gamepad` or `keyboard_mouse`. Absent in the Hajime config, where the `.get` default selects `gamepad` |
| `actions.mappings` | the `keyboard_mouse` branch of `step()` (lines 186-208) and the `keyboard_mouse` recording branches | **ignored in gamepad mode**: `step()` dispatches on hard-coded indices 0-17, so the `vg_code` names (`DS4_BUTTON_CROSS`, …) in the Hajime table are inert |
| `TRAINING_CONFIG.max_trajectories` | **read by nothing** | the recorder hard-codes `max_traj=10` (`record_trajectories.py:237`) |
| `TRAINING_CONFIG.batch_size` / `.epochs` / `.learning_rate` | argparse *defaults* only (`train_agent.py:389-391`) | overridden by `--batch` / `--epochs` / `--lr` |
| `TRAINING_CONFIG.window_size` | **read by nothing** | `TemporalAttentionLSTM` hard-codes `window_size = 10` |
| `TRAINING_CONFIG.dagger_iterations` | **read by nothing** | the DAgger path is a stub (§9) |
| `TRAINING_CONFIG.demo_path` / `.model_path` / `.train_path` | **read by nothing** | the scripts hard-code `./demos/`, `./models/`, `./models/imitation/bc_logs/` |
| `INPUT_CONFIG.deadzone`, `.camera_sensitivity`, `.input_delay` | **read by nothing** | `INPUT_CONFIG` is not imported by any module in the repository (verified by grep) |
| `INPUT_CONFIG.aggressiveness` | **read by nothing** | `run_ai.py:180` looks up `GAME_CONFIG.get("aggressiveness", 1.0)` — a *top-level* key that neither config defines. See §8 |

The minimum viable edit set is `process_name`, `window_offset`, `capture.internal_*` (leave at 128
unless you also touch the encoders), `actions.num_actions`, `actions.input_mode`, `actions.mappings`.

## 3. Recording demonstrations

### 3.1 Protocol

Start the game or emulator **first** (otherwise the environment waits 120 s and then degrades silently —
[SETUP.md §7](SETUP.md#7-game-and-emulator-prerequisites)), then `python record_trajectories.py`. An
854×480 pygame window titled `AI Agent Capture` appears with the upscaled 128×128 capture, an FPS
readout, `Demos: n/10`, a `RECORDING`/`IDLE` status and the prompt
`[K] Toggle Record   [ESC] Save & Exit`. Key bindings, exactly as implemented:

| Input | Effect | Line |
|---|---|---|
| `K` | toggle recording on/off (0.3 s debounce) | 101-104 |
| `ESC` | break the loop and exit | 96-98 |
| window close button | same as `ESC` (`pygame.QUIT`) | 92-94 |
| arrow keys `↑ ↓ ← →` / `i`, `o`, `p` | set bits 0-3 / bits 4-6 (gamepad mode only) | 122-128 |
| physical pad on XInput slot 0 | D-pad masks `0x0001/2/4/8` → bits 0-3; left stick beyond ±16 000 → bits 0-3; `0x1000/0x2000/0x4000` (A/B/X) → bits 4-6 | 131-153 |
| configured keys / mouse buttons | in `keyboard_mouse` mode, `keyboard.is_pressed(mapping["key"])` and `mouse.is_pressed(mapping["button"])` per mapping index | 111-119 |

**`ESC` does not save.** The comment at line 223 (`# Saving now happens in main()'s finally block`) is
wrong: `main()`'s `finally` only prints and calls `pygame.quit()`. A trajectory is written **only** when
recording is stopped with `K` (lines 197-213) — press `K` before `ESC` or the buffer is lost.

### 3.2 What the recorder writes

`demos/demo_<n>_<YYYYMMDD>_<HHMMSS>.pt` — `<n>` is the zero-based index of the saved trajectory, the
timestamp is `time.strftime("%Y%m%d_%H%M%S")` at the moment of the stop. The payload is `th.save([traj])`:
a **list holding one `imitation.data.types.Trajectory`** dataclass with fields `obs`, `acts`, `infos`
(`None`) and `terminal` (`False`), so loading one requires the `imitation` package. Shapes measured
directly from the five corpus files:

| File | `obs` | `acts` |
|---|---|---|
| `demo_0_20260523_221206.pt` | `(6302, 4, 128, 128)` uint8 | `(6301, 18)` float32 |
| `demo_1_20260523_221830.pt` | `(7069, 4, 128, 128)` uint8 | `(7068, 18)` float32 |
| `demo_0_20260523_224241.pt` | `(8150, 4, 128, 128)` uint8 | `(8149, 7)` float32 |
| `demo_1_20260523_224751.pt` | `(5644, 4, 128, 128)` uint8 | `(5643, 7)` float32 |
| `demo_0_20260619_103226.pt` (hajime) | `(6151, 4, 128, 128)` uint8 | `(6150, 18)` float32 |

`obs` has **one row more than `acts`** by design: the first frame is appended twice (lines 192-194) so
that the final observation is the outcome of the final action. Stacking is 4-deep, done *at capture time*.

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

**Dead bits 7-17 are not an operator failure — they are unreachable by code.** In gamepad mode the
recorder writes only indices 0-6 (`record_trajectories.py:122-153`; the comment
`# Other buttons can be mapped here` at line 155 marks the gap), so no amount of recording will set bit 7
(LT), 8 (RT), 9 (L3) or 10-17 (right stick). Fixing them means extending the recorder to read
`bLeftTrigger` / `bRightTrigger`, the `0x0200`/`0x0400`/`0x0800`/`0x1000` masks and `sThumbRX/RY` into
those indices, then re-recording — or switching to `keyboard_mouse` with 9 mappings, where every bit *is*
reachable.

**Class imbalance is fixable by recording behaviour:**

1. **Many short clips, not one long session.** `K`, play one specific situation for 60-90 s, `K`. Each
   stop writes a separate file, so the shuffler sees independent trajectories rather than one
   autocorrelated block; the loop caps at 10 files per process, so run several sessions.
2. **Deliberately actuate the rare bits** — 5 and 6 fire below 1 %. A defence-only clip, a whiffed-combo
   clip, a cornered clip and an approach clip beat five more neutral clips.
3. **Cover states, not frames:** more frames of the same kind teach nothing new; the missing *action
   combinations* do.
4. **Keep widths consistent** — do not reconfigure `num_actions` between recordings into the same
   `demos/` folder; two of the four generic files are 7-wide because of exactly that (§12).
5. Verify with `python train_agent.py --epochs 1 --batch 32 --device cpu`, whose
   `Action distribution (0-N):` histogram prints the marginal firing rate per bit.

## 4. Training

### 4.1 `train_agent.py` — the primary BC entry point

| Flag | Type | Default | Effect |
|---|---|---|---|
| `--epochs` | int | `TRAINING_CONFIG["epochs"]` = 100 | BC epochs, passed to `bc_trainer.train(n_epochs=…)` |
| `--batch` | int | `TRAINING_CONFIG["batch_size"]` = 384 | demonstration minibatch size |
| `--lr` | float | `TRAINING_CONFIG["learning_rate"]` = 1e-4 | constant schedule (`optimizer_kwargs={"lr": …}`) |
| `--dagger` | flag | off | accepted; calls `run_dagger_iteration()`, which prints a line and executes `pass` (§9) |
| `--device` | str | `cuda` | torch device. **No automatic CPU fallback** — see SETUP.md §2 |
| `--model_path` | str | `None` | warm start from an existing `.zip` (§7) |

Backbone is `ActorCriticCnnPolicy` (NatureCNN) unless a per-architecture script is used. Writes
`models/bc_policy.zip` and `models/imitation/bc_logs/progress.csv`; logs to experiment
`Hajime_no_Ippo_Imitation_Learning` / run `BC_Training` — **`generic_agent` logs to that Hajime-named
experiment too** (`train_agent.py:406`).

### 4.2 Per-architecture scripts

| Script | Extractor | Flags | Experiment / run name | Checkpoint written |
|---|---|---|---|---|
| `train_agent.py` | NatureCNN | `--epochs --batch --lr --dagger --device --model_path` | `Hajime_no_Ippo_Imitation_Learning` / `BC_Training` | `models/bc_policy.zip` |
| `train_agent_lstm.py` | `TemporalAttentionLSTM` | `--epochs --batch --lr --device --model_path` | `Generic_Agent_LSTM_IL` / `BC_LSTM_Training` | `models/bc_policy_lstm.zip` |
| `train_agent_transformer.py` | `VisionTransformerExtractor` | `--epochs --batch --lr --dagger --device --model_path` | `Generic_Agent_Transformer_IL` / `BC_Training_Transformer` | `models/bc_policy_transformer.zip` |
| `train_agent_impoola.py` | `ImpoolaCNNExtractor` | `--epochs --batch --lr --device` (defaults 10/384/1e-4) | `Model_Comparison` / `Impoola_CNN` | `models/ImpoolaCNN_policy.zip` |
| `compare_models.py` | all six | `--epochs --batch --lr --device --only-new` (defaults 10/384/1e-4) | `Model_Comparison` / one run per name + `Comparison_Summary` | `models/<Name>_policy.zip` |
| `train_imiation.py` | NatureCNN | **no argparse at all** | `Hajime_no_Ippo_Imitation_Learning` / `BC_Training_Simple` | `models/bc_policy.zip` |

`Generic_Agent_LSTM_IL` and `Generic_Agent_Transformer_IL` have no directory under
`generic_agent/mlruns/`, so **neither script has ever completed an MLflow-initialised run** here; those
architectures were benchmarked only through `compare_models.py`.

### 4.3 Budgets, and the evidence for them

```powershell
# Architecture study (regenerates README §10.1 and models/comparison_results.md)
python compare_models.py --epochs 10 --batch 384 --lr 1e-4
# A converged single model
python train_agent.py --epochs 300 --batch 384 --lr 1e-4 --device cuda
```

10 epochs is a **screening budget, not a training budget**. The benchmark stops every encoder between
2.90 and 3.48, while the one retained long run on the same corpus (`models/imitation/bc_logs/progress.csv`,
42 logged rows spanning `bc/epoch` 0 → 292) falls from `bc/loss` 4.851 to **1.330** and lifts
`bc/prob_true_act` from 0.0078 to **0.483**. Ten epochs is ≈ 3 % of what this corpus can absorb and the
loss is still falling steeply at epoch 10, so a 10-epoch figure does not describe a converged model, and
encoders separated by <0.13 there are not separated at all. Always report the budget with the loss (§13).

## 5. Reading the metrics

All `bc/*` keys are emitted by `imitation`'s own loss calculator
(`imitation/algorithms/bc.py:126-155`) into the SB3 logger and forwarded to MLflow by each script's
`MLflowOutputFormat` (`train_agent.py:72-83`). `BC.train` dumps every `log_interval` batches (default
500, `bc.py:388`), which is why the retained `progress.csv` rows sit at `bc/batch` 0, 500, 1000, ….

| Metric | Definition in code | Informative? |
|---|---|---|
| `bc/neglogp` | `-log_prob.mean()`, where `log_prob` is `Bernoulli.log_prob(acts).sum(dim=1)` → **mean over the batch of the sum over bits**, in nats | yes — the actual BC objective |
| `bc/prob_true_act` | `exp(log_prob).mean()` → mean joint probability assigned to the demonstrated action vector | **the most interpretable fit metric**; 1/2ⁿ for a uniform predictor, 0.48 at epoch 292 |
| `bc/loss` | `neglogp + ent_loss + l2_loss` | yes, but it is a composite; watch it drift from `neglogp` only if entropy changes fast |
| `bc/entropy` | `Bernoulli.entropy().sum(dim=1).mean()` → summed over bits. ≈ 12.5 nats for 18 unbiased bits; observed 4.85 → 1.43 | yes — a healthy collapse. If it reaches ~0 early the policy has collapsed to a marginal predictor |
| `bc/ent_loss` | `-1e-3 × bc/entropy` (`ent_weight` default `1e-3`, `bc.py:286`) | no, derivative |
| `bc/l2_norm` | `½ Σ w²` over all policy parameters | mildly — a rising curve with flat loss suggests divergence |
| `bc/l2_loss` | `l2_weight × l2_norm` with `l2_weight` default **`0.0`** (`bc.py:287`). Verified identically `0.0` in all 42 logged rows | **dead** |
| `bc/epoch`, `bc/batch`, `bc/samples_so_far` | loop counters written by `bc.py:231-234`; `samples_so_far` is the cumulative samples consumed (`bc.py:163-167`), so its final logged value 7 872 384 is exactly `20 501 × 384` | bookkeeping |
| `batch_size` | the logger's own key, equal to `--batch` (`bc.py:231`) | bookkeeping |
| `final_loss` | `compare_models.py:208`: the **last value** of `bc/loss` seen by `MetricCapture`, i.e. one minibatch, not an epoch average | yes, but noisy by construction |
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
mlflow ui --backend-store-uri file:../mlruns
```

Open <http://localhost:5000>, from `<pkg>/notebooks`: both the `--backend-store-uri` value and the
scripts' own `mlflow.set_tracking_uri("file:../mlruns")` resolve against the CWD (verified from
`mlruns/*/meta.yaml`, whose `artifact_location` reads
`file:///D:/Imitation-player/generic_agent/notebooks/../mlruns/...`), so each package has its **own**
store and no shared UI. Experiments present on disk:

| Experiment | Package | Runs |
|---|---|---|
| `Model_Comparison` | `generic_agent` | 26 |
| `Hajime_no_Ippo_Imitation_Learning` | `generic_agent` | 5 |
| `Hajime_no_Ippo_Imitation_Learning` | `hajime_agent` | 1 |
| `Default` | `hajime_agent` | 0 |

Filter per-batch curves by prefix: `compare_models.py` namespaces them `<Model>/bc/…` (line 175), e.g.
`Impoola_CNN/bc/loss` against `ResNet18/bc/loss`. The store is **git-ignored** (`.gitignore`: `mlruns/`,
`*/mlruns/`), so these runs exist only on the workstation that produced them.

## 7. Transfer / warm start

```powershell
python train_agent.py --epochs 50 --batch 384 --lr 1e-4 --model_path ./models/bc_policy.zip
```

`train_agent.py:307-312`: if the path exists, `ActorCriticCnnPolicy.load(model_path, device=…)` is
called and the instance is handed to `BC(policy=…)`. Three consequences. (1) The **architecture is
inherited from the checkpoint**, not chosen by the script — there is no extractor flag. (2) Optimiser
state is not restored and `lr_schedule` is rebuilt, so the rate restarts at `--lr`. (3) The run
**overwrites its own input**, since the output path is fixed at `models/bc_policy.zip`; copy the seed
first if you want to keep it. A missing path prints `[!] Warning: model '<path>' not found. Training from
scratch.` and continues rather than aborting. `train_agent_lstm.py` and `train_agent_transformer.py` also
accept `--model_path`; `train_agent_impoola.py` and `compare_models.py` do not.

## 8. Deploying a policy

```powershell
python run_ai.py                 # CNN policy from models/bc_policy.zip
python run_ai_lstm.py            # models/bc_policy_lstm.zip
python run_ai_transformer.py     # models/bc_policy_transformer.zip
```

| Script | Model it reads | Keys | Declared `MAX_FPS` |
|---|---|---|---|
| `run_ai.py` | `models/bc_policy.zip`, else `get_last_index("./models/steps", "bc_policy", ".zip")` | `K` toggle AI/manual, `ESC` quit, `↑ ↓ ← → i o p` for manual play in gamepad mode | generic 120 / hajime 30 |
| `run_ai_lstm.py` | `models/bc_policy_lstm.zip` **only** (no fallback) | as above | 30 / 30 |
| `run_ai_transformer.py` | `models/bc_policy_transformer.zip` **only** | as above | 120 / 120 |

Verified facts about the deploy path:

- **`MAX_FPS` is never read.** Every occurrence in the tree is a declaration; no `time.sleep` derives
  from it and `loop_start` (line 153) is assigned and unused. The loop is **unpaced**, so the 120-vs-30
  difference between packages is documentary, not behavioural; throughput is set by capture latency plus
  `policy.predict`.
- **The `aggressiveness` branch is inert.** `run_ai.py:180` reads `GAME_CONFIG.get("aggressiveness",
  1.0)`, but the value `2.0` lives in `INPUT_CONFIG` (generic config line 126), which no module imports,
  so the default `1.0` always applies and execution takes the plain
  `policy.predict(obs, deterministic=False)` branch. When it *does* fire (after a local edit adding a
  top-level key) it is not logit sharpening: `run_ai.py:193-202` takes `sigmoid(action_net(...))`,
  multiplies only bits whose `mappings[i]["type"] == "mouse_button"` by the factor, clamps to 1.0 and
  samples `(rand(len(probs)) < probs)`. The branch exists in `generic_agent`'s `run_ai.py` and
  `run_dagger.py`; `hajime_agent/notebooks/run_ai.py` has none (its LSTM/Transformer runners do).
- **Nothing deploys the benchmarked encoders.** `compare_models.py` writes `models/<Name>_policy.zip`
  and `train_agent_impoola.py` writes `models/ImpoolaCNN_policy.zip`; no `run_ai*` script reads either.
  Copy first: `Copy-Item models/ImpoolaCNN_policy.zip models/bc_policy.zip`.
- Recurrent models are auto-detected (`hasattr(policy.features_extractor, "lstm")`) and wrapped in
  `LSTMWrapper`, whose `reset()` fires only on `K` re-activation, since `done` is never `True`.

## 9. DAgger (interactive correction)

```powershell
python run_dagger.py
```

Bindings (`run_dagger.py:148-161`; overlay prompt at line 85 reads
`[K] Rec  [L] Toggle AI/Human  [ESC] Exit`):

| Key | Effect |
|---|---|
| `K` | start/stop a correction trajectory (0.3 s debounce). On the transition to recording, control is forced to the AI and the LSTM is reset |
| `L` | **toggle authority**: `action_from_ai = not action_from_ai`; the overlay prints `CONTROL: AI` / `CONTROL: HUMAN` |
| `ESC` | exit. As in the recorder, an in-progress trajectory is **not** flushed — press `K` first |

Worked example: press `K` and let the policy play; the moment it errs press `L` to take over, play the
recovery, press `L` to hand back, then `K` to close the episode. Output is
`demos/dagger_demo_<n>_<YYYYMMDD>_<HHMMSS>.pt` — one `Trajectory` in a list, format as §3.2 — and
**only frames recorded while control was human carry human actions**; AI frames are logged with the
policy's own sampled actions. Loop cap `MAX_TRAJ = 10`.

The `dagger_demo_*` prefix matches the `demo*.pt` glob used by every loader, so the next training run
picks the corrections up automatically. In gamepad mode, though, the human correction vector is read
**only** from the keyboard (`↑ ↓ ← → i o p`, lines 182-188): unlike the recorder, `run_dagger.py` does
not poll XInput, so a physical pad cannot be used to correct.

**Honest status of the loop.** The aggregation-and-retrain half does not exist. Both stubs are verbatim:
`train_agent.run_dagger_iteration()` (`train_agent.py:367-374`) is a signature, one `print`, and `pass`;
`train_imiation.dagger_iteration()` (`train_imiation.py:155-163`) is a docstring,
`# TODO: a full DAgger implementation requires a human-in-the-loop`, and `pass`. `--dagger` therefore
changes nothing: DAgger is a **manual two-command workflow** (`run_dagger.py`, then re-run
`train_agent.py`, which retrains over all of `./demos/`). No DAgger iteration has ever been executed —
the store contains no run attributable to it.

## 10. GAIL (adversarial imitation)

```powershell
python train_gail.py --timesteps 100000
```

| Flag | Default | Effect |
|---|---|---|
| `--timesteps` | 100 000 | total *environment* steps for the PPO generator |
| `--model_path` | `None` | `PPO.load(model_path, env=…)` as the generator seed |

Fixed hyperparameters (`train_gail.py:141-145`, `159-161`): PPO generator with `batch_size=64`,
`learning_rate=3e-4`, `n_steps=1024`, `ent_coef=0.01`, `gamma=0.99`, `ActorCriticCnnPolicy`;
`BasicRewardNet` discriminator with `RunningNorm`; `demo_batch_size=64`,
`gen_replay_buffer_capacity=2048`, `n_disc_updates_per_round=4`. Logs to
`models/imitation/gail_logs/progress.csv` and to experiment `Hajime_no_Ippo_Imitation_Learning` / run
`GAIL_Run`; saves `models/gail_policy.zip` (the *policy*, not the full PPO).

**Warnings.** `GAILTrainer._create_env()` hard-sets `train_config["dummy"] = False` (line 114) because
the dummy environment cannot even `reset()` — `_get_observation` reads `self.camera`, an attribute the
dummy branch never creates. **GAIL therefore requires the game running, the window found and the capture
path working: it is impossible headless and impossible without ViGEmBus.** The 2048-step generator buffer
against 100 000 requested timesteps keeps ~2 % of the trajectory in the replay window, and the reward is
discriminator-derived on a 27 k-frame corpus. Finally: **GAIL has never completed a run in this project
and never started one on this machine** — no `GAIL_Run` and no `gail_logs` directory exist in either
store (verified by grep over `*/mlruns`). Read this whole section as a description of code, not a result.

## 11. Benchmark

```powershell
python compare_models.py --epochs 10 --batch 384 --lr 1e-4   # full six-model study
python compare_models.py --only-new                           # two models, four stored baselines
```

A full run trains all six encoders in sequence, saves `models/<Name>_policy.zip` per architecture, opens
one MLflow run each plus a `Comparison_Summary` run, then calls `save_and_update_results()`, which
**rewrites two files**: (1) the repository-root `../../README.md` — the table between
`<!-- BENCHMARK_START -->` and `<!-- BENCHMARK_END -->` is replaced and the line after the
`**Conclusion**:` label regenerated from the measured values; nothing else is touched, so keep both
markers intact; (2) `./models/comparison_results.md`, regenerated wholesale including the
*Experiment Configuration* block recording the epoch/batch/lr/device actually used.

**`--only-new` warning.** It trains only `Impala_CNN` and `ResNet18` and substitutes **hard-coded
constants** for the other four (`compare_models.py:436-473`); it does **not** read the MLflow store. The
table therefore mixes fresh measurements with numbers frozen from an earlier session, and since both
packages carry the same dict, a `--only-new` run inside `hajime_agent` would print `generic_agent`'s
corpus numbers. Treat it as authoritative only after a **full** run. A related trap:
`train_agent_impoola.py:149-167` embeds the NatureCNN, CNN_LSTM and ViT baselines too and rewrites
`comparison_results.md` as a four-column table whenever it runs.

## 12. Known broken things

| # | Defect | Where |
|---|---|---|
| 1 | `train_imiation.py` is misspelled in both packages **and** globs `demos*.pt` while the recorder writes `demo*.pt`: it loads zero demonstrations, prints `No trajectories to train!`, returns from `train()`, and then reaches `mlflow.log_artifact('./models/bc_policy.zip')` — which can only succeed off a checkpoint left by a *different* script | `train_imiation.py:68, 107-109, 198` vs `record_trajectories.py:205` |
| 2 | Mixed demonstration widths are silently reconciled — 18-wide truncated, 7-wide zero-padded to the configured 9, unlogged. The published benchmark therefore ran on a 9-bit label space assembled from Hajime recordings | `train_agent.py:155-177`, `compare_models.py:122-127` |
| 3 | A demo file that raises while loading is swallowed by `except Exception` + `continue`, so a truncated corpus trains normally and only a console line distinguishes it | `train_agent.py:183-185`, `compare_models.py:132-133`, `train_gail.py:92-94` |
| 4 | A failed capture returns the **previous** frame (black if none), with no counter and no warning — indistinguishable from a legitimately black screen | `game_env.py:311-313` |
| 5 | Bits 7-17 are unreachable: the gamepad recording branch writes only 0-6, so the declared 18-bit space can never be fully populated by data | `record_trajectories.py:122-153` |
| 6 | `ESC` does not flush the in-progress trajectory, despite the overlay's `[ESC] Save & Exit` and an inline comment claiming a `finally`-block save that does not exist | `record_trajectories.py:96-98, 222-223, 243-250`; `run_dagger.py:160-161, 266-269` |
| 7 | `aggressiveness` is looked up on the wrong dictionary, so the inference-time sharpening branch never executes | `run_ai.py:180` vs `generic_agent/config/game_config.py:126` |
| 8 | `MAX_FPS` is declared and never read; `loop_start` is assigned and never read — the deploy loops are unpaced | `run_ai.py:32, 153` |
| 9 | A dummy environment cannot `reset()` or `step()`: `self.camera`, `self.mss_sct` and `self.gamepad` are created only *after* the `dummy` early return, so any stepping path raises `AttributeError` | `game_env.py:102-108` |
| 10 | In gamepad mode `step()` indexes `actions[10:18]` unconditionally, so a `gamepad` configuration with `num_actions < 18` raises `IndexError` | `game_env.py:248-262` |
| 11 | `train_imiation.py` omits `VecFrameStack`: its observation space is `(1, 128, 128)` while demonstrations are stored `(N, 4, 128, 128)` | `train_imiation.py:57-64` |
| 12 | `hajime_agent/notebooks/train_agent.py` has no width reconciliation, so the mixed-width corpus that `generic_agent` survives would fail there | verified: 0 occurrences of `expected_num_actions` |

**Correction to a widely-repeated claim.** README §12.6 and ARCHITECTURE.md §5 say `get_last_index`
"compares checkpoint suffixes lexicographically, so `bc_policy9.zip` outranks `bc_policy10.zip`". That is
**not what the code does**: `utils/utils.py:14-39` extracts the digits and folds
`max(last_index, int(m.group(1)))` — a numeric maximum, so no such inversion occurs. The real defect is
different: the directory it is asked to scan (`models/steps/`, from `STEPS_PATH = MODEL_PATH + "steps"`)
is created by no script, so the fallback always returns `-1` and the checkpoint-numbering feature is
dead; only the fixed-name `bc_policy.zip` branch ever resolves.

## 13. Reproducibility checklist

Quote a number from this project only if you can state all of the following.

1. **Package and CWD** — `generic_agent` or `hajime_agent`, run from `<pkg>/notebooks`.
2. **Commit and interpreter** — `git rev-parse HEAD`; `python -c "import sys,torch; print(sys.version,
   torch.__version__, torch.version.cuda)"`.
3. **Corpus** — the files matched by `demo*.pt`, their frame counts, their original widths and the
   `num_actions` they were reconciled to (item 2 of §12 makes this non-obvious); keep the printed
   action-distribution histogram from a `--epochs 1` pass.
4. **Budget and device** — epochs, batch, lr, `cuda`/`cpu` and the GPU model. A 10-epoch loss is not a
   converged loss (§4.3), and `training_time_s` is meaningless without the hardware stated.
5. **Seed status** — only the demonstration shuffler is seeded (`np.random.default_rng(seed=42)`). No
   `torch.manual_seed`, `cudnn.deterministic` or `torch.use_deterministic_algorithms` call exists in the
   tree, so initialisation and CUDA kernel order are uncontrolled and the same command will not
   reproduce a loss to the third decimal.
6. **Metric identity and its limits** — the key name (`bc/loss` vs `final_loss` vs `bc/neglogp`), that
   it is summed over `num_actions` bits, that it is in-sample, and that no held-out split, accuracy,
   rollout or success rate exists behind it. The figure ranks optimisation ease and nothing else.
7. **Artefacts** — attach `models/comparison_results.md`, the MLflow run id and
   `models/imitation/bc_logs/progress.csv`, noting that demonstrations, checkpoints and `mlruns/` are all
   git-ignored and therefore unobtainable from the repository.
