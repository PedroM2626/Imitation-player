# Experiments

> Per-script protocol, complete CLI reference, the exact definition of every logged metric, and the
> procedure for adding a new architecture. Interpretation of what was produced lives in
> [RESULTS.md](RESULTS.md); the reasons the numbers are weak live in [LIMITATIONS.md](LIMITATIONS.md).

## 1. Global experiment conventions

| Convention | Value | Set by |
|---|---|---|
| Invocation | every entry point is a module invocation from the **repository root**: `python -m agent.cli.<name>` | `agent/cli/common.py` puts the repository root on `sys.path`; no script depends on the current working directory any more |
| Demonstration source | `runs/<profile>/demos/`, glob `demo*.pt` | `agent/utils/paths.demos_dir`, derived from the profile (`demo_path` no longer exists as a config field) |
| Checkpoint destination | `runs/<profile>/models/` | `agent/utils/paths.models_dir` |
| MLflow tracking URI | `file:runs/<profile>/mlruns` | `agent/utils/tracking.configure_store` |
| Device | `cuda` when `th.cuda.is_available()`, else `cpu`; an explicit `--device` overrides either | `agent/cli/common.resolve_device` (36-44) |
| Objective | sigmoid-bernoulli BC NLL, **summed over bits**, in **nats** | `imitation.BC` |
| RNG | `np.random.default_rng(seed=42)` — in every training run, not only the benchmark | `SEED` in `agent/cli/train.py:34`, `agent/cli/train_gail.py:36` |
| Torch seeding | **none** — no `torch.manual_seed`, no `cudnn.deterministic` | — |

Because every path is derived from the selected profile rather than the working directory, the runs root
can be relocated wholesale with `IMITATION_RUNS` or `--runs-root`, and two profiles never share a
demonstration folder.

Because the loss is a *sum* over bits, absolute loss values are only comparable between runs that share
`num_actions`. The published benchmark uses 9; the Hajime configuration declares 18. Their loss scales
are not interchangeable, and this is why §DATA 4's marginal baseline must be recomputed per width.

## 2. Script reference

### 2.1 `agent/cli/record.py` — demonstration capture
| | |
|---|---|
| Algorithm | none (data collection) |
| CLI | `--profile`, `--runs-root`, `--max-trajectories` (default: the profile's `recording.max_trajectories`) |
| Env mode | live (`dummy = False`) |
| Keys | `K` start/stop recording; `ESC`, the window close button and Ctrl+C all flush the in-progress buffer and exit; input is read per the profile's `actions.input_mode`, either the `keyboard_mouse` mapping table or the physical pad with keyboard stand-ins |
| Output | `runs/<profile>/demos/demo_<n>_<YYYYMMDD>_<HHMMSS>.pt` |
| Specification | [DATA.md §2](DATA.md#2-recording-protocol) |

### 2.2 `agent/cli/train.py` — behavioural cloning, the single entry point
| Flag | Default | Effect |
|---|---|---|
| `--epochs` | `TRAINING_CONFIG["epochs"]` (100) | BC epochs |
| `--batch` | `TRAINING_CONFIG["batch_size"]` (384) | demonstration minibatch size |
| `--lr` | `TRAINING_CONFIG["learning_rate"]` (1e-4) | constant schedule (`lr_schedule=lambda _: lr`) |
| `--arch` | `naturecnn` | feature extractor, chosen from the registry in `agent/utils/architectures.py` |
| `--device` | `None` → resolved at runtime (`cuda` when available, else `cpu`) | torch device |
| `--model_path` | `None` | checkpoint to continue training from; a missing path aborts instead of silently starting from scratch |

Backbone `ActorCriticCnnPolicy` (NatureCNN encoder) unless the selected architecture supplies an
extractor. A new run logs to experiment `f"{profile}_imitation_bc"` (for example
`hajime_ippo_imitation_bc`) under a run named after `--arch`, and saves
`runs/<profile>/models/bc_policy.zip` for `naturecnn` or `<Prefix>_policy.zip` for the registered
architectures. Every run also records what it trained on — frames, trajectories, action width, distinct
joint actions and the marginal/uniform reference losses — via `agent/utils/tracking.log_dataset()`, and
`--width-policy strict|coerce` overrides the profile's demonstration-width policy. The four
near-duplicate `train_agent*.py` scripts of the old layout, and the inert `--dagger` flag they accepted,
are gone. The
historical runs retained in `runs/hajime_ippo/mlruns/` are the ones that all reported into
`Hajime_no_Ippo_Imitation_Learning`, including those trained on the generic reconciliation of the corpus.

### 2.3 `--arch` variants: `lstm`, `transformer`, `impoola` (`impala` and `resnet18` also registered)
One BC loop with a `policy_kwargs = {"features_extractor_class": …}` override per architecture.

| `--arch` | Extractor | Extra constructor defaults | Experiment | Run name |
|---|---|---|---|---|
| `lstm` | `TemporalAttentionLSTM` | `features_dim=512`, `lstm_hidden_size=256`, `lstm_num_layers=2`, `window_size` from `TRAINING_CONFIG["window_size"]` | `hajime_ippo_imitation_bc` | `lstm` |
| `transformer` | `VisionTransformerExtractor` | `embed_dim=256`, `patch_size=16`, `num_heads=4`, `num_layers=4`, `features_dim=512` | `hajime_ippo_imitation_bc` | `transformer` |
| `impoola` | `ImpoolaCNNExtractor` | `features_dim=512`, `channels_list=[32,64,128]` | `hajime_ippo_imitation_bc` | `impoola` |

CLI for all of them: `--arch --epochs --batch --lr --device --model_path --width-policy`. The old
scripts disagreed about the default budget — `train_agent_impoola.py` used 10 epochs while the others
used the config's 100 — and that inconsistency is gone: every architecture takes the same
`TRAINING_CONFIG` defaults, so a bare invocation is comparable across architectures.

The `--dagger` flag these scripts accepted was a `pass` stub; it no longer exists. The DAgger loop lives
in `agent/cli/dagger.py` (§2.8), which calls this trainer itself after each collection round.

### 2.4 `train_imiation.py` — removed
This entry point is gone, together with its misspelled filename: the minimal BC loop it held is folded
into `agent/cli/train.py`, which globs `demo*.pt` (the pattern the recorder writes) rather than the
`demos*.pt` that never matched anything, and which takes real argparse flags instead of none. Its
`dagger_iteration()` `pass`-with-`TODO` body is likewise gone, replaced by `agent/cli/dagger.py`. No
script in the tree references the name any more.

### 2.5 `agent/cli/train_gail.py` — adversarial imitation
| Flag | Default |
|---|---|
| `--timesteps` | 100,000 |
| `--model_path` | `None` |

Fixed internals, now exposed as flags (`--demo-batch-size`, `--gen-batch-capacity`,
`--disc-updates-per-round`): `imitation.GAIL` with `BasicRewardNet` discriminator and a PPO generator
(`batch_size=64`, `learning_rate=3e-4`, `n_steps=1024`, `ent_coef=0.01`, `gamma=0.99`);
`demo_batch_size=64`, generator replay capacity 2048, 4 discriminator updates per round; observations
normalised with `RunningNorm`. Forces `dummy = False`, so it requires the live game and therefore Windows.
New runs use experiment `f"{profile}_gail"` (e.g. `hajime_ippo_gail`). **Never completed a run** — no
`GAIL_Run` exists in the tracking store
([RESULTS.md §8](RESULTS.md#8-evidence-gaps-to-close-in-priority-order)).

### 2.6 `agent/cli/benchmark.py` — the benchmark harness
| Flag | Default | Effect |
|---|---|---|
| `--epochs` | 10 | applied to every architecture |
| `--batch` | 384 | |
| `--lr` | 1e-4 | |
| `--device` | `None` → resolved at runtime | |
| `--archs` | all six | subset trained now; the remaining columns are read back from the MLflow store instead of from hard-coded constants |

Sequence per architecture: load all demos → build the dummy env + wrappers → construct
`ActorCriticCnnPolicy` with the extractor (or pass `policy=None` for NatureCNN) → `BC(...)` →
`train(n_epochs, progress_bar=True)` → save `runs/<profile>/models/<Name>_policy.zip` → report the four
summary metrics (`final_loss`, `training_time_s`, `num_params`, `model_size_mb`), which reach MLflow only
through `agent.cli.train.main`'s own run. Then `regenerate_readme()` and `write_report()` regenerate
`runs/<profile>/models/comparison_results.md` **and rewrites the marked region of `README.md`**. See §5.

### 2.7 `agent/cli/deploy.py` — deployment
| Flag | Default | Effect |
|---|---|---|
| `--arch` | `naturecnn` | selects the checkpoint prefix to resolve (`bc_policy`, `bc_policy_lstm`, `bc_policy_transformer`, `ImpoolaCNN_policy`, …) |
| `--model` | `None` | explicit checkpoint path, bypassing resolution |
| `--max-steps` | `None` | stop after this many policy steps |
| `--start-manual` | off | begin with the human in control |

Loads the newest matching checkpoint via `resolve_checkpoint` (final `<prefix>.zip`, else the highest
numbered one), then loops `policy.predict(obs)` → `env.step(...)`. Keys: `K` toggles AI against human,
`ESC` quits; manual play reads the same `HumanInput` the recorder uses. Inference pacing is now
**enforced**: the loop sleeps out the remainder of the `deploy.fps` budget, so the declared rate is
behavioural rather than documentary. The `aggressiveness` knob is read from
`GAME_CONFIG["deploy"]["aggressiveness"]`, where the profile actually defines it, and applies to the
indices named in `deploy.attack_buttons`, so the sharpening path is reachable by editing the profile
([ARCHITECTURE.md §5](ARCHITECTURE.md#5-inference-path-agentclideploypy)).
**This script logs nothing**: no metric, no MLflow run, no artefact. It counts the steps it took and
prints the total on exit, but deployment remains otherwise unaudited and unmeasured.

### 2.8 `agent/cli/dagger.py` — interactive correction
| Flag | Default | Effect |
|---|---|---|
| `--arch` | `naturecnn` | policy to collect corrections for |
| `--rounds` | `TRAINING_CONFIG["dagger_iterations"]` (3) | collect-then-retrain iterations |
| `--seconds` | 120 | collection time per round |
| `--epochs` / `--batch` / `--lr` | `None` → `TRAINING_CONFIG` | passed through to the retrain call |
| `--collect-only` | off | stop after collection, do not retrain |

The policy acts; holding `L` transfers authority to the human, and releasing it flushes the recorded
segment to `runs/<profile>/demos/demo_dagger_<profile>_<YYYYMMDD>_<HHMMSS>.pt`. Unlike the old script,
this one drives the aggregation half of the loop as well: after each round it re-reads the corpus, prints
the new marginal baseline, and calls `agent.cli.train.main` to retrain over all of `demos/` — so the
former "`pass` stub" situation described in §2.2/§2.4 no longer exists.

## 3. Metric definitions

Emitted per gradient step by `imitation`'s logger and forwarded to MLflow under the prefix
`<Model>/` (`tracking.MLflowOutputFormat(prefix=f"{arch.name}/")`, `agent/utils/tracking.py`), so a run's
tree looks like `metrics/NatureCNN/bc/loss`, `metrics/NatureCNN/bc/entropy`, …

| Key | Definition | Informative? |
|---|---|---|
| `bc/loss` | sigmoid-bernoulli NLL, **summed over action bits**, nats | yes — the primary objective |
| `bc/neglogp` | negative log-probability of the demonstrated action | essentially `bc/loss` without the bonus term |
| `bc/prob_true_act` | `exp(−neglogp)`: joint probability of the human's exact action vector | **the most interpretable number in the store** |
| `bc/entropy` | mean policy entropy over the Bernoulli bits | yes — a collapse monitor |
| `bc/ent_loss` | entropy bonus term (negative; `ent_coef = 0` by default) | no, only tracks the coefficient |
| `bc/l2_norm` | gradient L2 norm | yes — divergence/instability detector |
| `bc/l2_loss` | weighted L2 penalty | **dead: identically `0.0` in every retained log**; new runs exclude it from the store altogether (`tracking.DEFAULT_SKIP_KEYS`), because `l2_weight` defaults to `0.0` |
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
# from the repository root
mlflow ui --backend-store-uri file:runs/hajime_ippo/mlruns      # http://localhost:5000
```

The tracking URIs the scripts set are derived from the profile, so one store per profile replaces the two
per-package stores; `--runs-root`/`IMITATION_RUNS` relocate them. Experiments present in
`runs/hajime_ippo/mlruns/`: `Model_Comparison` (26 runs), `Hajime_no_Ippo_Imitation_Learning` (6 runs
carried over from the generic store, 1 from the hajime one — the two entries keep the names they were
created under), `Default` (empty). Runs from here on land in derived names such as
`hajime_ippo_imitation_bc` and `hajime_ippo_gail`. The store is git-ignored, so a fresh clone has
nothing to show — see [README §15.2](../README.md#152-what-is-and-is-not-versioned).

## 5. Modifying `README.md` automatically

`python -m agent.cli.benchmark` rewrites this repository's top-level README on every run:

1. It regenerates the table between `<!-- BENCHMARK_START -->` and `<!-- BENCHMARK_END -->`, in a fixed
   column order: NatureCNN, CNN+LSTM+Attention, ViT, Impoola-CNN, Impala-CNN, ResNet-18.
2. If the string `**Conclusion**:` appears in the README, it replaces that entire paragraph with a
   sentence **derived from the measured values** (lowest-loss model and most efficient model, with loss,
   MB and parameter figures). The hand-written conclusion the old harness injected — which contradicted
   the table it sat under — is gone; if no loss/parameter pair can be measured, the generator declines to
   write one and leaves the README paragraph as it is. Keep that marker present exactly once or the
   replacement silently no-ops; keep it to a single paragraph, since the reconstruction preserves
   everything after the conclusion's first line.
3. Anything outside those two regions is untouched.

There are no fallback baseline constants any more. An architecture not selected with `--archs` is read
back from the MLflow store and rendered from its own recorded metrics; an architecture with no run in the
store renders as `-` with a `Runs available` count of 0, and is never substituted from a constant. The
previous design did exactly that: the `main()` dict's Impala-CNN and ResNet-18 entries held values from
*different* runs of the same configuration (2.945390/406.6 s and 3.018901/391.1 s), which is how the
generated report and the README disagreed. Two lessons: never let a report carry numbers it did not
measure, and a fallback table that must be hand-updated will drift.

## 6. Adding a new architecture

1. Implement `class MyExtractor(BaseFeaturesExtractor)` in `agent/utils/`, taking
   `(observation_space, features_dim=512, **kwargs)` and mapping `(B, 4, 128, 128) → (B, features_dim)`.
   Keep it **stateless** — do not replicate `TemporalAttentionLSTM`'s cross-call buffer
   ([ARCHITECTURE.md §4.5](ARCHITECTURE.md#45-temporalattentionlstm-agentutilstemporallstmpy28)).
2. Add one entry to `ARCHITECTURES` in `agent/utils/architectures.py` with its `name`,
   `checkpoint_prefix`, `factory` and constructor `kwargs`; the trainer, the deployer and the benchmark all
   read that registry. To appear as a benchmark column, add the key to `COLUMN_ORDER` and `DISPLAY` in
   `agent/cli/benchmark.py`.
3. A per-bit marginal baseline is computed and logged for you: `agent.utils.demos.summarise()` produces
   `marginal_baseline_nats` and `agent.utils.tracking.log_dataset()` writes it into the run, and
   `agent.cli.train` prints the run's margin against it. A model above `Σ H(m_i)` has still learned the
   prior, not a policy
   ([DATA.md §4](DATA.md#4-action-distribution), [RESULTS.md §5](RESULTS.md#5-trivial-baselines-what-the-reported-losses-actually-mean)).
4. Either run the full sweep so every column is measured in the same pass, or pass `--archs` and accept
   that unmeasured columns are filled from the store — or left as `-` when the store has nothing.
5. Report ≥5 seeds and a mean ± SD. Single runs on this corpus move by up to 0.117 nats
   ([RESULTS.md §4](RESULTS.md#4-run-to-run-dispersion)).

## 7. Protocol deficits to fix before any result here is citable

In order of how much each invalidates current claims:

1. **No held-out set.** Add a split over demonstrations (leave-one-session-out is available today with
   four generic files) and report per-bit accuracy, per-bit F1 and joint exact-match accuracy on it.
2. **No marginal baseline in the historical runs.** `Σ H(m_i)` is logged per run from now on, but the
   retained benchmark runs were recorded without it, and §DATA 4 shows every published run is worse than
   this trivial predictor.
3. **No seed control.** Seed torch, enable `cudnn.deterministic`, and run repeated seeds; the current
   dispersion data (§RESULTS 4) is incidental, which is worse than being planned.
4. **No closed-loop metric.** Blocked on the environment's absent reward/terminal signal
   ([README §5.3](../README.md#53-reward-termination-and-the-deployment-loop)).
5. **Budget too short to discriminate.** 10 epochs; the 300-epoch reference run is 1.28 nats better than
   the marginal bound, so the discriminating regime is somewhere between 10 and 300 epochs and has not
   been sampled.
6. **Dataset size is now recorded, but not checked.** Every run logs its frame and trajectory counts
   (§DATA 6.2), so a silently truncated corpus can be spotted after the fact; no run asserts the count it
   expected, and an unreadable demo is only caught because `load_demos` raises.
