# Experiments

> Per-script protocol, complete CLI reference, the exact definition of every logged metric, and the
> procedure for adding a new architecture. Interpretation of what was produced lives in
> [RESULTS.md](RESULTS.md); the reasons the numbers are weak live in [LIMITATIONS.md](LIMITATIONS.md).

## 1. Global experiment conventions

| Convention | Value | Set by |
|---|---|---|
| Working directory | every script must be run **from `<pkg>/notebooks/`** | `sys.path.insert(0, "..")` / `"../utils"` at the top of each script |
| Demonstration source | `./demos/`, glob `demo*.pt` | hard-coded, *not* `TRAINING_CONFIG["demo_path"]` |
| Checkpoint destination | `./models/` | hard-coded |
| MLflow tracking URI | `file:../mlruns` | hard-coded in every script |
| Device | `cuda`, with a runtime fallback to `cpu` if `th.cuda.is_available()` is false | `compare_models.py:421-426` |
| Objective | sigmoid-bernoulli BC NLL, **summed over bits**, in **nats** | `imitation.BC` |
| RNG | `np.random.default_rng(seed=42)` (benchmark only) | `compare_models.py:169` |
| Torch seeding | **none** — no `torch.manual_seed`, no `cudnn.deterministic` | — |

Because the loss is a *sum* over bits, absolute loss values are only comparable between runs that share
`num_actions`. The published benchmark uses 9; the Hajime configuration declares 18. Their loss scales
are not interchangeable, and this is why §DATA 4's marginal baseline must be recomputed per width.

## 2. Script reference

### 2.1 `record_trajectories.py` — demonstration capture
| | |
|---|---|
| Algorithm | none (data collection) |
| CLI | none; constants at the top of the file |
| Env mode | live (`dummy = False`) |
| Keys | `K` start/stop recording, `ESC` save and quit; the generic variant adds a keyboard/mouse mapping path |
| Output | `demos/demo_<n>_<YYYYMMDD>_<HHMMSS>.pt` |
| Specification | [DATA.md §2](DATA.md#2-recording-protocol) |

### 2.2 `train_agent.py` — behavioural cloning, primary script
| Flag | Default | Effect |
|---|---|---|
| `--epochs` | `TRAINING_CONFIG["epochs"]` (100) | BC epochs |
| `--batch` | `TRAINING_CONFIG["batch_size"]` (384) | demonstration minibatch size |
| `--lr` | `TRAINING_CONFIG["learning_rate"]` (1e-4) | constant schedule (`lr_schedule=lambda _: lr`) |
| `--dagger` | off | accepted; calls `run_dagger_iteration()`, which is a `pass` stub |
| `--device` | `cuda` | torch device |
| `--model_path` | `None` | directory or zip of a pretrained policy to continue training from |

Backbone `ActorCriticCnnPolicy` (NatureCNN encoder) unless `policy_kwargs` supplies an extractor.
Logs to experiment `Hajime_no_Ippo_Imitation_Learning`, run `BC_Training`, and saves
`models/bc_policy.zip`. Note that the **generic package also writes to the Hajime-named experiment**
(`generic_agent/notebooks/train_agent.py:406`), which is why the Hajime-titled experiment contains runs
trained on the generic reconciliation of the corpus.

### 2.3 `train_agent_lstm.py`, `train_agent_transformer.py`, `train_agent_impoola.py`
Same BC loop with a `policy_kwargs = {"features_extractor_class": …}` override.

| Script | Extractor | Extra constructor defaults | Experiment | Run name |
|---|---|---|---|---|
| `train_agent_lstm.py` | `TemporalAttentionLSTM` | `features_dim=512`, `lstm_hidden_size=256`, `lstm_num_layers=2` | `Generic_Agent_LSTM_IL` | `BC_LSTM_Training` |
| `train_agent_transformer.py` | `VisionTransformerExtractor` | `embed_dim=256`, `patch_size=16`, `num_heads=4`, `num_layers=4`, `features_dim=512` | `Generic_Agent_Transformer_IL` | `BC_Training_Transformer` |
| `train_agent_impoola.py` | `ImpoolaCNNExtractor` | `features_dim=512`, `channels_list=[32,64,128]` | `Model_Comparison` | `Impoola_CNN` |

CLI for all three: `--epochs --batch --lr --device`. `train_agent_impoola.py` defaults to **10 epochs**
while the other two default to the config's 100 — an inconsistency that makes a bare invocation of the
Impoola script incomparable to a bare invocation of the others.

`train_agent_transformer.py` and `train_agent_lstm.py` also accept `--dagger`, likewise inert.

### 2.4 `train_imiation.py` — minimal BC
No CLI at all. Experiment `Hajime_no_Ippo_Imitation_Learning`, run `BC_Training_Simple`.
**Broken in two ways:** the filename is a misspelling of *imitation*, and it globs `demos*.pt` while the
recorder writes `demo*.pt`, so it loads zero demonstrations and fails at dataset construction. Its
`dagger_iteration()` is a `pass` with a `TODO`. Do not use it; it is retained only because notebooks
reference the name.

### 2.5 `train_gail.py` — adversarial imitation
| Flag | Default |
|---|---|
| `--timesteps` | 100,000 |
| `--model_path` | `None` |

Fixed internals: `imitation.GAIL` with `BasicRewardNet` discriminator and a PPO generator
(`batch_size=64`, `learning_rate=3e-4`, `n_steps=1024`, `ent_coef=0.01`, `gamma=0.99`);
`demo_batch_size=64`, generator replay capacity 2048, 4 discriminator updates per round; observations
normalised with `RunningNorm`. Forces `dummy = False`, so it requires the live game. Experiment
`Hajime_no_Ippo_Imitation_Learning`, run `GAIL_Run`. **Never completed a run** — no `GAIL_Run` exists in
the tracking store ([RESULTS.md §8](RESULTS.md#8-evidence-gaps-to-close-in-priority-order)).

### 2.6 `compare_models.py` — the benchmark harness
| Flag | Default | Effect |
|---|---|---|
| `--epochs` | 10 | applied to every architecture |
| `--batch` | 384 | |
| `--lr` | 1e-4 | |
| `--device` | `cuda` | |
| `--only-new` | off | trains Impala-CNN and ResNet-18 only; substitutes **hard-coded** constants for the other four |

Sequence per architecture: load all demos → build the dummy env + wrappers → construct
`ActorCriticCnnPolicy` with the extractor (or pass `policy=None` for NatureCNN) → `BC(...)` →
`train(n_epochs, progress_bar=True)` → save `models/<Name>_policy.zip` → log the four summary metrics.
Then `save_and_update_results()` regenerates `models/comparison_results.md` **and rewrites the marked
region of `README.md`**. See §5.

### 2.7 `run_ai.py`, `run_ai_lstm.py`, `run_ai_transformer.py` — deployment
No CLI. Load the newest matching checkpoint via `get_last_index`, then loop `policy.predict(obs)` →
`env.step(...)`. Keys: `K` toggles AI against human, `ESC` quits. Inference pacing is a script constant:
`MAX_FPS` 120 in `generic_agent`, 30 in `hajime_agent`. `generic_agent/notebooks/run_ai.py` additionally
implements the `aggressiveness` branch (apply `action_net` → sigmoid → raise to a power → threshold).
**These scripts log nothing**: no metric, no MLflow run, no counter of frames or actions. Deployment is
therefore unaudited and unmeasured.

### 2.8 `run_dagger.py` — interactive correction
No CLI. The policy acts; `L` transfers authority to the human; the human's inputs during those windows
are appended as a new demonstration. Only the *collection* half of DAgger exists — see §2.2/§2.4 stubs.

## 3. Metric definitions

Emitted per gradient step by `imitation`'s logger and forwarded to MLflow under the prefix
`<Model>/` (`MLflowOutputFormat(prefix=f"{name}/")`), so a run's tree looks like
`metrics/NatureCNN/bc/loss`, `metrics/NatureCNN/bc/entropy`, …

| Key | Definition | Informative? |
|---|---|---|
| `bc/loss` | sigmoid-bernoulli NLL, **summed over action bits**, nats | yes — the primary objective |
| `bc/neglogp` | negative log-probability of the demonstrated action | essentially `bc/loss` without the bonus term |
| `bc/prob_true_act` | `exp(−neglogp)`: joint probability of the human's exact action vector | **the most interpretable number in the store** |
| `bc/entropy` | mean policy entropy over the Bernoulli bits | yes — a collapse monitor |
| `bc/ent_loss` | entropy bonus term (negative; `ent_coef = 0` by default) | no, only tracks the coefficient |
| `bc/l2_norm` | gradient L2 norm | yes — divergence/instability detector |
| `bc/l2_loss` | weighted L2 penalty | **dead: identically `0.0` in every retained log** |
| `bc/epoch` | epoch index at the time of logging | — |
| `bc/batch` | cumulative batch counter | — |
| `bc/samples_so_far` | cumulative demonstration samples consumed | needed to reconstruct the effective dataset size |

Summary metrics logged once per run by the scripts themselves:

| Key | Definition |
|---|---|
| `final_loss` | the last `bc/loss` value captured by `MetricCapture` |
| `training_time_s` | wall time of `bc_trainer.train()` only, excluding data loading |
| `num_params` | `sum(p.numel() for p in policy.parameters() if p.requires_grad)` — **the whole actor-critic policy, not the encoder alone** |
| `model_size_mb` | `os.path.getsize(zip) / 1024²` — includes the optimiser state and the normaliser buffers SB3 stores in the archive |

Caveats to keep in mind when reading any of these:

- **No accuracy metric is computed anywhere**, per-bit or joint. Earlier documentation claimed MLflow
  logged accuracy; it does not. Per-bit accuracy would be the natural companion to a summed NLL and is
  a five-line addition in `MetricCapture`.
- `num_params` counts the critic and the MLP head as well, so the encoder-to-encoder parameter
  differences are somewhat diluted; report encoder-only counts when comparing architectures.
- `model_size_mb` measures a *serialised archive*, not weights only, which is why a 4.20 MB Impoola
  checkpoint and a 16.33 MB NatureCNN checkpoint are not 4× different in floating-point weight volume
  alone.
- Logging density: every completed benchmark run wrote **two** `bc/loss` points over 10 epochs, because
  `imitation`'s default log interval is coarse relative to the ~700 gradient steps in a run. Curves in
  the UI are effectively straight lines between two dots.

## 4. Reading MLflow

```bash
cd generic_agent/notebooks
mlflow ui --backend-store-uri file:../mlruns      # http://localhost:5000
```

Experiments present: `Model_Comparison` (26 runs), `Hajime_no_Ippo_Imitation_Learning` (6 runs in the
generic store, 1 in the hajime store), `Default` (empty). The store is git-ignored, so a fresh clone has
nothing to show — see [README §15.2](../README.md#152-what-is-and-is-not-versioned).

## 5. Modifying `README.md` automatically

`compare_models.py` rewrites this repository's top-level README on every run:

1. It regenerates the table between `<!-- BENCHMARK_START -->` and `<!-- BENCHMARK_END -->`, in a fixed
   column order: NatureCNN, CNN+LSTM+Attention, ViT, Impoola-CNN, Impala-CNN, ResNet-18.
2. If the string `**Conclusion**:` appears in the README, it replaces that entire paragraph with a
   sentence **derived from the measured values** (lowest-loss model and most efficient model, with loss,
   MB and parameter figures). Keep that marker present exactly once or the replacement silently
   no-ops; keep it to a single paragraph, since the reconstruction preserves everything after the
   conclusion's first line.
3. Anything outside those two regions is untouched.

The fallback baseline constants in `main()` are used whenever an architecture is not retrained. They
were realigned with the completed MLflow runs in this revision; the Impala-CNN and ResNet-18 entries
previously held values from *different* runs of the same configuration (2.945390/406.6 s and
3.018901/391.1 s), which is how the generated report and the README disagreed. Two lessons: never let
a report carry numbers it did not measure, and a fallback table that must be hand-updated will drift.

## 6. Adding a new architecture

1. Implement `class MyExtractor(BaseFeaturesExtractor)` in `generic_agent/utils/`, taking
   `(observation_space, features_dim=512, **kwargs)` and mapping `(B, 4, 128, 128) → (B, features_dim)`.
   Keep it **stateless** — do not replicate `TemporalAttentionLSTM`'s cross-call buffer
   ([ARCHITECTURE.md §4.5](ARCHITECTURE.md#45-temporalattentionlstm-utilsgame_envpy379)).
2. Register it in `compare_models.py`'s architecture list with its `policy_kwargs`, and add the key to
   `model_keys` and `display_names`.
3. **Compute a per-bit marginal baseline for the corpus and log it** before judging the run; a model
   above `Σ H(m_i)` has learned the prior, not a policy
   ([DATA.md §4](DATA.md#4-action-distribution), [RESULTS.md §5](RESULTS.md#5-trivial-baselines-what-the-reported-losses-actually-mean)).
4. Add a fallback entry to the `baselines` dict, or run the full sweep so every column is measured in
   the same pass.
5. Report ≥5 seeds and a mean ± SD. Single runs on this corpus move by up to 0.117 nats
   ([RESULTS.md §4](RESULTS.md#4-run-to-run-dispersion)).

## 7. Protocol deficits to fix before any result here is citable

In order of how much each invalidates current claims:

1. **No held-out set.** Add a split over demonstrations (leave-one-session-out is available today with
   four generic files) and report per-bit accuracy, per-bit F1 and joint exact-match accuracy on it.
2. **No marginal baseline.** Log `Σ H(m_i)` per run; without it the loss numbers have no reference point,
   and §DATA 4 shows every published run is worse than this trivial predictor.
3. **No seed control.** Seed torch, enable `cudnn.deterministic`, and run repeated seeds; the current
   dispersion data (§RESULTS 4) is incidental, which is worse than being planned.
4. **No closed-loop metric.** Blocked on the environment's absent reward/terminal signal
   ([README §5.3](../README.md#53-reward-termination-and-the-deployment-loop)).
5. **Budget too short to discriminate.** 10 epochs; the 300-epoch reference run is 1.28 nats better than
   the marginal bound, so the discriminating regime is somewhere between 10 and 300 epochs and has not
   been sampled.
6. **Nothing logs the effective dataset size into MLflow**, so no run records how many frames it
   actually trained on — and §DATA 6.2 shows files can be skipped silently.
