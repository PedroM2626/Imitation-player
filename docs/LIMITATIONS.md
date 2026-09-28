# Limitations and open work

> A complete audit of what this repository does and does not establish, and of what is broken.
> Every item carries a status, because the point of this document is to say what has actually been dealt
> with and what has merely been written down.
>
> **Status key**
> `OPEN` — a real limit on the evidence, unchanged.
> `MITIGATED` — the tooling now prevents the failure; the existing published numbers still carry it.
> `FIXED` — resolved in this revision, with the test or file that pins it.
>
> Cross-references: [README §12](../README.md#12-threats-to-validity) ·
> [RESULTS.md](RESULTS.md) · [DATA.md](DATA.md) · [EXPERIMENTS.md](EXPERIMENTS.md) ·
> [ARCHITECTURE.md](ARCHITECTURE.md) · [SETUP.md](SETUP.md)

## Part I — Limitations of the evidence

### L1 · OPEN — No held-out evaluation exists anywhere
**Evidence.** Nothing in `agent/` splits the demonstrations, computes per-bit accuracy, runs an
evaluation rollout or compares against the human demonstrator. Every script trains on the whole
`runs/<profile>/demos/` directory.
**Consequence.** No generalisation number can be reported. "Result" has so far meant "training loss".
**Fix.** Leave-one-file-out over the corpus (four usable generic sessions exist today), reporting
per-bit accuracy, per-bit F1 and joint exact-match on the held-out session.

### L2 · MITIGATED — No reference baseline, and the published numbers lose to a trivial one
**Evidence.** The sigmoid-bernoulli BC loss is a sum over bits in nats, so a frame-independent per-bit
marginal predictor scores `Σᵢ H(mᵢ)` = **2.6066** nats on the 9-bit reconciled labels the benchmark used.
Published range: 2.9017 – 3.4763 ([RESULTS.md §5](RESULTS.md#5-trivial-baselines-what-the-reported-losses-actually-mean)).
**Consequence.** At a 10-epoch budget no architecture has been shown to condition on the screen at all;
every cross-architecture difference in README §10.1 is a difference in speed of approach to the prior.
**What changed.** `agent/utils/demos.summarise()` computes the marginal and uniform references from the
corpus, `agent.cli.train` prints them before training, `agent.utils.tracking.log_dataset()` writes them
into every MLflow run, and the generated benchmark table carries a "Margin over the marginal baseline"
row. A future run cannot be reported without its reference point. The historical numbers are unchanged.

### L3 · OPEN — Run-to-run variance is as large as the effect being measured
**Evidence.** ResNet-18 at 10 epochs: 2.9017 / 2.9639 / 3.0189 (range 0.117). Impala-CNN: 2.9454 – 3.0298
(range 0.084). NatureCNN: 3.4763 / 3.5817. Three nominally identical 100-epoch BC runs differ by 0.75
nats and by a factor of 3.8 in `prob_true_act`. The published ranking of four encoders spans 0.13.
**Consequence.** The ranking of the top four is unsupported. Only "NatureCNN is clearly worse" survives.
**Fix.** Seed torch, set `cudnn.deterministic`, ≥5 seeds per architecture, mean ± SD, paired comparison.
Note that `agent/cli/train.py` still seeds only the NumPy generator; nothing seeds torch.

### L4 · OPEN — The training budget sits on the steepest part of the curve
**Evidence.** 10 epochs → 2.90–3.48 nats; 300 epochs on the same corpus → 1.330 nats with
`prob_true_act` 0.483; an abandoned ResNet-18 run reached 2.851 before interruption.
**Consequence.** Encoders differ chiefly in convergence rate, so the benchmark measures that rate rather
than capacity.
**Fix.** Sweep 10 / 30 / 100 / 300 epochs and report where the ordering changes.

### L5 · MITIGATED — Most of the action space is unused, partly because it was unrecordable
**Evidence.** Bits 7–17 have marginal exactly 0.0 in every existing file. Only 45 distinct joint actions
occur in 27,161 generic-pool frames and 24 in the Hajime session, of `2¹⁸`.
**Root cause, now fixed.** The old gamepad recorder wrote **only indices 0–6**
(`record_trajectories.py:122-155`, with `# Other buttons can be mapped here` marking the gap), so
triggers, stick press and the camera axes could not be demonstrated no matter how the operator played.
`agent/utils/input_map.py` now maps `bLeftTrigger`/`bRightTrigger`, both thumb presses and
`sThumbRX/RY` onto bits 7–17, and `agent/utils/emission.py` drives the device from the same table, so
capture and replay agree.
**What is still open.** No existing demonstration exercises those bits, so no result involves them, and
11 heads in every published model have a provably constant target.

### L6 · OPEN — Class imbalance and idling dominate the labels
**Evidence.** Generic pool: up 0.495, down 0.375, right 0.207 versus bit 5 at 0.0062 and bit 6 at 0.0038.
Hajime session: the modal joint action is the all-zero "do nothing" vector, 51.3 % of frames; top five
cover 80 %.
**Consequence.** An unweighted product-of-sigmoids learns to idle; low activity is predicted correctly
without perception.
**Fix.** Subsample idle frames or weight the per-bit loss, and report accuracy conditioned on non-idle
frames alongside the unconditional figure. `demos.summarise()` already reports `zero_action_share`.

### L7 · MITIGATED — Data provenance was absent
**Evidence.** Demonstration files carry no recording metadata: `infos` is `None`, `terminal` is `False`,
and no capture geometry, game build, `num_actions` in force or operator identity is stored. MLflow logged
no frame count, so a run that silently lost files was indistinguishable from one that did not.
**What changed.** Every run now logs `profile`, `corpus_frames`, `corpus_trajectories`, `num_actions`,
`distinct_joint_actions`, both baselines, the seed, the width policy and the device
(`tracking.log_dataset`). The corpus itself is still not self-describing.
**Fix.** Write a sidecar JSON per demo (or populate `infos`) and checksum the corpus per run.

### L8 · MITIGATED — The benchmark corpus does not match the profile that reports it
**Evidence.** The four files under the old `generic_agent/notebooks/demos/` are Hajime recordings — two
18-wide, two 7-wide — while that package's configuration targeted Roblox with 9 keyboard/mouse actions.
Loaders coerced everything to 9.
**Consequence.** The published study trained on a 9-bit space assembled from another game's data.
**What changed.** The corpus is now filed under the profile it belongs to (`runs/hajime_ippo/`), the
`roblox` profile is explicitly labelled as never trained, and the 9-bit coercion is no longer the
default. **Still open:** the numbers in README §10.1 are 9-bit figures and cannot be regenerated by a
strict 18-bit run; reproducing them needs `--width-policy coerce` with a 9-action profile.

### L9 · MITIGATED — Silent degradation paths
**Evidence and what changed.**
- Capture returned the previous (or black) frame with no counter → now counted in
  `GenericGameEnv.dropped_frames`.
- Demo loading swallowed per-file failures in `except Exception` → a corrupt file is now a hard error
  listing every failure (`tests/test_demos.py::test_corrupt_file_raises_instead_of_shrinking_the_corpus`).
- Action-width reconciliation was silent → `width_policy = "strict"` raises naming the file and both
  widths; `coerce` warns per file (`tests/test_train_smoke.py`).
- A missing window warned and left a broken environment alive → `WindowNotFoundError`
  (`tests/test_env.py::test_window_timeout_raises_rather_than_degrading`).
**What is still open.** `dropped_frames` is counted but not surfaced in the progress bar or logged to
MLflow; a run trained on frames with many stale grabs looks normal.

### L10 · OPEN — Retracted architecture claims lack measurements
**Evidence.** README §7.7 asserts Swin and ConvNeXt are too slow and VRAM-hungry (≈28 M parameters, "VRAM
thrashing", ≈40 h for 10 epochs). The store holds 1 `Swin_Transformer`, 2 `ConvNeXt` and 6 `Vision_Mamba`
runs, all interrupted at the first logged batch, with no memory or throughput measurement anywhere
(no `torch.cuda.max_memory_allocated`, no `nvidia-smi` call — verified by grep).
**Consequence.** There is no evidence about those architectures in either direction. The withdrawal is an
engineering judgement that this repository previously presented as if measured.
**Fix.** Re-measure peak VRAM and steps/second, or state the withdrawal as unquantified (the README now
does). `Vision_Mamba` additionally needs its source recovered — only `mamba_architectures.cpython-311.pyc`
survives.

### L11 · FIXED — Documentation contradicted the data it described
**Evidence.** Earlier README revisions claimed MLflow logged accuracy (it does not), asserted Impoola-CNN
had the lowest loss (ResNet-18 did, 2.90 vs 2.93), and the benchmark injected a hand-written conclusion
repeating that error while carrying hard-coded baselines that matched no completed run.
**Fix.** The conclusion is generated from the measurements and a regression test asserts it cannot name a
winner other than the minimum-loss architecture (`tests/test_benchmark_format.py::
test_conclusion_does_not_contradict_the_table`). Fallback columns are read from the store or rendered as
`-` with a zero run count. Every number in this documentation set was recomputed from the files.

### L12 · OPEN (deliberate) — The recurrent encoder violates batch independence
**Evidence.** `TemporalAttentionLSTM` keeps a `deque(maxlen=window_size)` of CNN features and a hidden
state across `forward()` calls; BC minibatches are shuffled, no episode boundary is defined, and
`reset_hidden()` is never called by a training loop. Gradients are detached on buffer entries and the
stacked sequence is re-flagged `requires_grad_(True)` in place, so only the most recent timestep trains
the CNN.
**Why it was not "fixed".** Changing the numerics would invalidate the CNN+LSTM row of the published
benchmark. The module was moved intact to `agent/utils/temporal_lstm.py`, its window length is now
configurable through `TRAINING_CONFIG.window_size` (default 10, the old hard-coded value), and the file's
docstring states the property.
**Proper fix.** Accept an explicit temporal window in the batch dimension, or feed contiguous episode
slices with `reset_hidden()` at boundaries — then re-measure that row.

## Part II — Component status

| # | Component | Status | Evidence | What changed / what remains |
|---|---|---|---|---|
| B1 | Evaluation | **OPEN** | see L1 | not built |
| B2 | DAgger re-aggregation | **FIXED** | `train_agent.run_dagger_iteration()` and `train_imiation.dagger_iteration()` were `pass`/`TODO` stubs | `agent/cli/dagger.py` runs collect → write → retrain for `--rounds` iterations, defaulting to `TRAINING_CONFIG.dagger_iterations`; never executed against a live game |
| B3 | GAIL | **OPEN** | no `GAIL_Run` in either store; needs a live window | moved to `agent/cli/train_gail.py`, hyperparameters exposed as flags, logs dataset provenance; still no completed run and no test |
| B4 | `train_imiation.py` | **FIXED** | misspelled in both packages and globbed `demos*.pt`, loading zero demonstrations | file deleted; `agent.cli.train --arch` replaces it |
| B5 | Deployment logging | **OPEN** | deploy loops record nothing | `agent/cli/deploy.py` prints control state and step count; no metrics persisted |
| B6 | `Dockerfile` | **MITIGATED, UNTESTED** | old image had no interpreter in the CUDA `-runtime` base and imported `vgamepad` at module scope | installs Python 3.11 + OpenCV libs, uses `requirements-docker.txt`, CPU one-epoch `CMD`; the image has never been built |
| B7 | Import portability | **FIXED** | `game_env.py` was unimportable off Windows, so even dummy training was | `agent/utils/windows.py` feature flags; `agent.utils.game_env` imports and dummy steps work anywhere — exercised by CI on `ubuntu-latest` |
| B8 | `Vision_Mamba` | **OPEN** | source deleted, only `__pycache__/*.pyc` remains | unreproducible |
| B9 | `SpatialAttention` | **FIXED** | dead class in `game_env.py` | deleted |
| B10 | Deployment checkpoint path | **FIXED** | searched `./models/steps`, which no script creates | `agent/utils/checkpoints.resolve_checkpoint` reads the profile's model directory; `tests/test_checkpoints.py` |
| B11 | Decorative config fields | **FIXED** | `capture.width/height`, `self.frame_time`, `MAX_FPS`, all of `INPUT_CONFIG`, `TRAINING_CONFIG.{window_size,dagger_iterations,demo_path,model_path,train_path,max_trajectories}` | every field is now consumed or deleted; the two meaningless width fields and three path fields were removed; `GenericGameEnv.step()` honours `target_fps` and `deploy.fps` is enforced. See README §5.5 |
| B12 | `actions.mappings` ignored in gamepad mode | **FIXED** | `step()` dispatched on hard-coded indices; `vg_code` names such as `DS4_BUTTON_CROSS` were never resolved | `agent/utils/emission.py` builds the device from the table, validates it against `num_actions`, and sums-and-clamps conflicting axis bits instead of last-index-wins. `tests/test_emission.py` |
| B13 | Hard-coded absolute paths | **FIXED** | `hajime_agent/config/game_config.py` embedded one user's RPCS3 and ROM paths | profiles set `exe_path`/`rom_path` to `None`; `agent/config/local.py` (git-ignored, `local.example.py` template) supplies them, and `tests/test_config.py` fails if a path reappears in a profile |
| B14 | Duplicate packages | **FIXED** | ten of eighteen modules byte-identical across two packages | collapsed into `agent/` with `config/profiles/`; one implementation of environment, emission, capture, training and deployment |
| B15 | Misleading MLflow experiment names | **FIXED** | the generic package wrote to `Hajime_no_Ippo_Imitation_Learning`; the Impoola script wrote to `Model_Comparison` | names derive from `f"{profile}_{purpose}"` (`tracking.experiment_name`). Historical runs keep their old experiment names — that is what the store contains |
| B16 | `bc/l2_loss` | **FIXED** | identically `0.0` in every retained log (`l2_weight` defaults to 0) | excluded from the store by `tracking.DEFAULT_SKIP_KEYS`; `bc/l2_norm` still logged |
| B17 | No tests, no linter, no CI | **FIXED** | none existed | 69 pytest tests in `tests/`; `.github/workflows/ci.yml` with a lint job (ruff check + `ruff format --check`) and a Linux test job on CPU torch including a one-epoch training step; `pyproject.toml` carries the ruff configuration and the codebase is formatted. **Still missing:** a coverage gate, and any test that needs a live game |
| B18 | No licence | **FIXED** | unlicensed public repository | MIT `LICENSE` added, with third-party terms listed in README §18 |
| B19 | CPU fallback unreachable | **FIXED** | `device = args.device or check_cuda()` never called `check_cuda()` because `--device` defaulted to `"cuda"` | `--device` defaults to unset and `common.resolve_device()` probes CUDA; covered by the CI training step running `--device cpu` |
| B20 | `inputs==0.5` | **MITIGATED** | imported by nothing; dead weight in the frozen environment | excluded from `requirements-minimal.txt` and `requirements-docker.txt`; retained in `requirements.txt`, which is kept as the verbatim record of the publishing environment |
| B21 | Recorder `ESC` discarded data | **FIXED** | the overlay read "`[ESC] Save & Exit`" and an inline comment claimed a `finally`-block save that did not exist; only the `K` stop-transition wrote a file | `TrajectoryRecorder.finish()` flushes on ESC, window-close and Ctrl+C, and the overlay is now accurate. `tests/test_recorder.py` (4 tests) |
| B22 | `run_dagger.py` exit path | **FIXED** | same data-loss pattern | `agent/cli/dagger.py` flushes its buffer in a `finally` |
| B23 | 11 action bits unrecordable | **FIXED** in tooling, **OPEN** in data | see L5 | `agent/utils/input_map.py` covers all 18 bits; no corpus exercises 7–17 |
| B24 | Deployment switches inert | **FIXED** | `MAX_FPS` declared and never read; `aggressiveness` looked up at the top level of `GAME_CONFIG` while defined in `INPUT_CONFIG`, so the sharpening path was dead | `deploy.fps` is enforced; the value moved to `GAME_CONFIG["deploy"]["aggressiveness"]` where the code reads it, and `PolicyRunner.sharpened()` implements it |
| B25 | Model artifacts in git without LFS | **FIXED** | a 17 MB `bc_policy.zip` sat in the object store | untracked, Git LFS configured for `*.zip` via `.gitattributes`, and the checkpoint published as a GitHub release asset with a documented `curl` fetch (README §15.2) |
| B26 | Frame-stacking skew between recorder and environment | **OPEN** | demos are stored pre-stacked `(N,4,128,128)` by the recorder, while the live environment declares `(128,128,1)` and is stacked only by `VecFrameStack`; shapes agree, code paths do not | nothing verifies that deployment observations are distributed like training ones; needs a held-out rollout (L1) |

## Part III — Prioritised open work

**P0 — required before any number here is citable.**
1. Held-out split and per-bit/joint metrics (L1).
2. Report margin over the marginal baseline — the tooling now logs it, so this is analysis, not code (L2).
3. Seed torch; ≥5 seeds; publish mean ± SD (L3).
4. Surface `dropped_frames` and a corpus checksum per run so silent data deficits are visible (L9).

**P1 — makes the existing claims meaningful.**
5. Budget sweep 10 / 30 / 100 / 300 epochs (L4).
6. Re-record exercising bits 7–17, or reduce the declared space; handle the idle-class imbalance (L5, L6).
7. Re-record under the `roblox` profile, or retire that profile as an example only (L8).
8. Close the recorder/deployer logging gap (B5).

**P2 — engineering hygiene, independent of the research.**
9. Add a coverage floor to CI (ruff, the lint job and the test job now exist — B17).
10. Build the Docker image once and record the result (B6).
11. Recover or reimplement `Vision_Mamba`, then measure it or delete the claim (B8, L10).
12. Populate `infos` / a sidecar manifest per demonstration (L7).

**P3 — structural.**
13. Make `TemporalAttentionLSTM` stateless or episode-aware, then re-measure its benchmark row (L12, B26).
14. Add a reward/terminal signal (HUD or referee reader) so episodic and closed-loop metrics become
    computable at all; until then no success-rate number can exist.
15. Version the corpus and store (LFS or a dataset DOI) so §10 is externally reproducible.

## Part IV — What can be stated honestly today

* A black-box, pixel-only screen-capture environment with virtual-gamepad and keyboard/mouse actuation
  works well enough to record human demonstrations and to train offline on them.
* The full pipeline — record → clone → track → deploy → correct → benchmark — is implemented, and the
  aggregation loop that used to be a stub now exists.
* Offline training is portable: it imports and runs on Linux without the capture stack, and CI proves it
  on every push.
* Among six encoders, SB3's default NatureCNN is *clearly* the worst in-sample (≈0.5 nats behind every
  custom encoder, far outside run-to-run spread).
* Replacing the flatten head with global average pooling cuts checkpoint size and parameter count by
  ~94 % while improving recorded loss and cutting training time by a third — a reproducible efficiency
  result (1.01 M parameters, 4.20 MB, 2.93 nats), regardless of whether that loss is competitive with a
  baseline.
* ViT is parameter-cheap (2.45 M) and compute-expensive (55 min vs 18 s): a quantified cost observation.
* **Nothing about playing ability has been demonstrated**, and at the published budget no encoder has
  been shown to beat a predictor that ignores the screen.
