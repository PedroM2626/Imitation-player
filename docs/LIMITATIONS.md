# Limitations and open work

> A complete account of what this repository does not yet establish, what is broken, and what has to be
> done. It is written to be read as a audit rather than as a disclaimer: each item states the evidence,
> the consequence for the claims, and the concrete change required.
>
> Cross-references: [README §12](../README.md#12-threats-to-validity) ·
> [RESULTS.md](RESULTS.md) · [DATA.md](DATA.md) · [EXPERIMENTS.md](EXPERIMENTS.md) ·
> [ARCHITECTURE.md](ARCHITECTURE.md)

## Part I — Limitations of the evidence

### L1. No held-out evaluation exists anywhere
**Evidence.** A repository-wide search for `success_rate`, `action_accuracy`, `evaluate`, `rollout`,
`validation`, `train_test_split` yields nothing but string-formatting helpers in `compare_models.py`.
Every script trains on the whole `./demos/` directory.
**Consequence.** No generalisation number can be reported. The word "result" in this project has so far
meant "training loss".
**Fix.** A split over demonstrations — leave-one-session-out is available today with the four generic
files — reporting per-bit accuracy, per-bit F1 and joint exact-match on the held-out session.

### L2. No reference baseline, and the published numbers lose to a trivial one
**Evidence.** The sigmoid-bernoulli BC loss is a sum over bits in nats, so the frame-independent
per-bit-marginal predictor scores `Σᵢ H(mᵢ)` = **2.6066** nats on the 9-bit reconciled labels used by the
benchmark. The best published model scores **2.9017**; the worst **3.4763**
([RESULTS.md §5](RESULTS.md#5-trivial-baselines-what-the-reported-losses-actually-mean)).
**Consequence.** At the 10-epoch budget no architecture has been shown to condition on the screen at
all. Every cross-architecture difference in README §10.1 is a difference in *speed of approach to the
action prior*.
**Fix.** Compute and log `Σ H(m_i)` per run and per split, and re-report the benchmark as margin over
that baseline. This is a few lines and should gate any future result claim.

### L3. Run-to-run variance is as large as the effect being measured
**Evidence.** Same-configuration repeats at 10 epochs: ResNet-18 2.9017 / 2.9639 / 3.0189 (range 0.117);
Impala-CNN 2.9454 / 3.0010 / 3.0276 / 3.0298 (range 0.084); NatureCNN 3.4763 / 3.5817 (range 0.105).
The published ranking of the four best encoders spans 2.90–3.03, i.e. 0.13. Three nominally identical
100-epoch BC runs differ by 0.75 nats (2.005 / 2.448 / 2.752) and by a factor of 3.8 in
`prob_true_act`.
**Consequence.** The ranking of the top four encoders is not supported. Only "NatureCNN is clearly worse"
survives, and it does so by a margin ~4× the largest spread.
**Fix.** ≥5 seeds per architecture, `torch.manual_seed` + `cudnn.deterministic`, mean ± SD, and a paired
test. Report parameter counts and wall time as deterministic quantities separately from loss.

### L4. The training budget sits on the steepest part of the curve
**Evidence.** Benchmark budget 10 epochs → 2.90–3.48 nats. A 300-epoch run on the same corpus reached
**1.330** nats with `prob_true_act` 0.483 ([RESULTS.md §6](RESULTS.md#6-the-budget-is-the-binding-constraint)).
An abandoned ResNet-18 run had already reached 2.851 before being interrupted.
**Consequence.** Encoders differ chiefly in convergence rate, so a budget that is too short measures
that rate rather than capacity. The ranking is a statement about early optimisation.
**Fix.** Sweep the budget (10 / 30 / 100 / 300 epochs) and report the point at which the ordering
changes, if it does.

### L5. The action space is mostly unused, so part of the "task" is fictitious
**Evidence.** Bits 7–17 (both triggers, stick press, and all eight camera flags) have marginal exactly
0.0000 in **every** file of both packages. Only 45 distinct joint actions occur in 27,161 generic frames
and 24 in 6,150 hajime frames, of `2¹⁸` possible.
**Consequence.** 11 output heads have a provably trivial target, which lowers every model's loss by an
architecture-independent constant and dilutes the parameter counts (`num_params` includes those heads).
More importantly, nothing in the corpus can teach camera control, blocking or trigger use — so the claim
that the agent "plays the game" is unsupported on its face.
**Fix.** Either record the missing behaviours, or reduce the declared action space to the bits in use and
re-derive the loss so the numbers remain comparable across studies.

### L6. Class imbalance and idling dominate the labels
**Evidence.** Generic pool marginals: up 0.495, down 0.375, right 0.207, action-4 0.125, but bit 5 0.0062
and bit 6 0.0038. In the hajime session the modal joint action is **the all-zero "do nothing" vector, at
51.3 % of frames**, and the top five cover 80 %.
**Consequence.** A product-of-sigmoids trained unweighted on this distribution learns to idle; low
activity is predicted correctly without any perception.
**Fix.** Subsample idle frames, or weight the per-bit loss, and report accuracy conditioned on
non-idle frames alongside the unconditional number.

### L7. Data provenance is absent
**Evidence.** Demonstration files carry no recording metadata (`infos` is a list of empty dicts): no
capture geometry, no game build, no `num_actions` in force, no operator, no timestamp correlation to the
config revision.
**Consequence.** The width discrepancy in L8 is only discoverable by opening the files, and no run can be
tied to the exact corpus that produced it. MLflow logs no frame count either, so a run trained on half
the data is indistinguishable from one trained on all of it.
**Fix.** Write a sidecar JSON per demo, populate `infos`, and log `total_frames` and a corpus checksum
per MLflow run.

### L8. The benchmark corpus does not match the package that reports it
**Evidence.** `generic_agent/notebooks/demos/` contains Hajime-era recordings — two files with 18-wide
action vectors and two with 7-wide — while `generic_agent/config/game_config.py` targets `RobloxPlayerBeta`
with 9 keyboard/mouse actions. The loaders truncate/pad all of them to 9 silently.
**Consequence.** The "generic" benchmark was run on Hajime data mapped into a 9-bit space assembled from
two recording regimes. The `generic_agent` label describes the code path (input emission), not the data.
**Fix.** Re-record under the generic configuration, or rename the packages so that the empirical claim
matches the title.

### L9. Silent degradation paths can change a result without an error
**Evidence.** Four independent cases: capture returns the previous (or a black) frame with no counter;
demo loading is wrapped in per-file `try/except Exception` that prints one line and continues; action
width reconciliation emits nothing; window lookup warns after 120 s and then leaves a permanently broken
environment object alive.
**Consequence.** In a study with no held-out metric, a data-integrity failure is indistinguishable from a
finding. Two of these paths can change the effective training set.
**Fix.** Fail loudly, or log the skipped files and frame counts into MLflow so the deficit is visible in
the run record.

### L10. Retracted claims lack measurements
**Evidence.** README §7.7 asserts Swin-Transformer and ConvNeXt are too slow and VRAM-hungry (≈28 M
parameters, "VRAM thrashing", ≈40 h for 10 epochs). The tracking store holds one `Swin_Transformer` run,
two `ConvNeXt` runs (one failed after 93.7 s) and six `Vision_Mamba` runs — all interrupted at the first
logged batch, none with a memory or throughput measurement.
**Consequence.** There is no evidence about those architectures at all, positive or negative. The
withdrawal is an engineering judgement, not a result, and the README previously presented it as if
measured.
**Fix.** Either re-measure (peak VRAM, steps/s, projected wall time at batch 384) and report it, or state
the withdrawal explicitly as unquantified.

### L11. Train/deploy skew in the frame-stack path
**Evidence.** Demos are stored pre-stacked `(N, 4, 128, 128)` by the recorder; the live environment
declares `(128, 128, 1)` and is stacked only by `VecFrameStack(n_stack=4)`. The two produce the same
tensor shape by different code.
**Consequence.** Nothing verifies that the deployment observations are distributed like the training
observations. A held-out evaluation would have caught this; there is none.
**Fix.** Assert shape and dtype equality between the recorder output and the wrapped env output in a
test, and evaluate on live rollouts.

### L12. The recurrent encoder violates batch independence
**Evidence.** `TemporalAttentionLSTM` keeps a `deque(maxlen=10)` of CNN features and a hidden state
across `forward()` calls; BC minibatches are shuffled, no episode boundary is defined, and
`reset_hidden()` is never called by a training loop. Gradients are detached on buffer entries and the
stacked sequence is re-flagged `requires_grad_(True)` in place, so only the most recent timestep trains
the CNN.
**Consequence.** Its output is a function of call history, which makes the reported CNN+LSTM loss
difficult to interpret as a per-example fit and makes the run non-reproducible under any change to batch
composition.
**Fix.** Make the extractor stateless (accept an explicit temporal window in the batch dimension), or feed
it contiguous episode slices with `reset_hidden()` at boundaries.

## Part II — Broken and unvalidated components

| # | Component | State | Evidence | Required change |
|---|---|---|---|---|
| B1 | Evaluation | **absent** | L1 | build a split + metrics module |
| B2 | DAgger re-aggregation | **stub** | `train_agent.run_dagger_iteration()` and `train_imiation.dagger_iteration()` are `pass`/`TODO`; no DAgger run in the store | implement the loop, or delete the flag and document the manual workflow |
| B3 | GAIL | **never completed a run** | no `GAIL_Run` in either store; forces `dummy=False` | run it, or mark it experimental |
| B4 | `train_imiation.py` | **broken** | globs `demos*.pt` while the recorder writes `demo*.pt` → loads zero demos; filename misspelled in both packages | fix glob and rename, or delete |
| B5 | Deployment scripts | **unaudited** | `run_ai*.py` log no metric, no counter, no MLflow run | log rollout length, action distribution, and toggle events |
| B6 | `Dockerfile` | **non-functional** | CUDA `-runtime` base has no Python/pip; `CMD` runs a training script; `sed` stripping leaves `pydirectinput`/`mouse` uninstalled while `game_env.py` imports `vgamepad` at module top | rewrite the image, and make `vgamepad`/`dxcam`/`pywin32` imports conditional so dummy-mode training is portable |
| B7 | `venv/` in this working copy | **broken** | points at `C:\Users\pedro\...\Python311\python.exe`, which does not exist on this machine | delete and recreate; it is git-ignored so it is a local problem only |
| B8 | `Vision_Mamba` | **unreproducible** | source module deleted; only `generic_agent/utils/__pycache__/mamba_architectures.cpython-311.pyc` survives | recover or reimplement, then re-measure |
| B9 | `SpatialAttention` | **dead code** | `game_env.py:368`, referenced by nothing | delete |
| B10 | `get_last_index` | **wrong ordering** | compares checkpoint suffixes lexicographically, so `bc_policy9.zip` outranks `bc_policy10.zip` | parse as int |
| B11 | Config fields | **decorative** | `TRAINING_CONFIG.window_size`, `dagger_iterations`, `demo_path`, `model_path`, `train_path` and all four `INPUT_CONFIG` fields are read by nothing; scripts hard-code `./demos/`, `./models/` | wire them up or remove them |
| B12 | `actions.mappings` in gamepad mode | **ignored** | `step()` dispatches on hard-coded indices; the declared `vg_code` names (`DS4_BUTTON_CROSS`, …) are never resolved, and the env presses `XUSB_GAMEPAD_A/B/X` | drive emission from the table |
| B13 | Hard-coded absolute paths | **unportable** | `hajime_agent/config/game_config.py` embeds a local RPCS3 executable path and ROM path; `compare_models.py` assumes `../../README.md` from cwd | require env vars or a local, git-ignored override file |
| B14 | Duplicate packages | **drift-prone** | 9 modules byte-identical across `hajime_agent/` and `generic_agent/`; the two `compare_models.py` copies carried different baseline constants (now identical) | collapse to one package + per-title config |
| B15 | MLflow experiment naming | **misleading** | `generic_agent/notebooks/train_agent.py` writes to experiment `Hajime_no_Ippo_Imitation_Learning`; `train_agent_impoola.py` writes to `Model_Comparison` | derive the experiment name from the package/title |
| B16 | `bc/l2_loss` | **dead metric** | identically `0.0` in every retained log | drop it, or enable the penalty term |
| B17 | No tests, no linter, no CI | **absent** | no test directory, no config for any linter, no workflow definitions | add unit tests for env shapes, the action mapping table, demo load/save round-trip, and `get_last_index` |
| B18 | No licence | **unspecified** | README §18 records the derivation from third-party work but no `LICENSE` file exists and no licence was chosen | choose and add one, or state explicitly that no rights are granted |
| B21 | Recorder `ESC` handling | **data loss** | the overlay reads `[ESC] Save & Exit` and an inline comment claims a `finally`-block save, but `ESC` only breaks the loop; a trajectory is written **only** by the `K` stop-transition (`record_trajectories.py:96-98`, `197-213`, `243-250`) | flush the buffer on exit, or relabel the key and the overlay |
| B22 | `run_dagger.py` exit path | same data-loss pattern as B21 | `run_dagger.py:160-161`, `266-269` | as above |
| B23 | Recorder action mapping | **11 bits unrecordable** | in gamepad mode the recorder writes only indices 0-6 (`record_trajectories.py:122-155`, with `# Other buttons can be mapped here` at 155); triggers, stick press and both camera axes are never captured, so the declared 18-bit space cannot be filled by any data this repository produces | map `bLeftTrigger`/`bRightTrigger`, the `0x0800`(Y)/`0x0400`/thumb masks and `sThumbRX/RY` to indices 7-17, or shrink `num_actions` to what is reachable |
| B19 | CPU fallback in `train_agent.py` / `train_agent_lstm.py` / `train_agent_transformer.py` | **unreachable** | `device = args.device or check_cuda()`, but `--device` defaults to the non-empty string `"cuda"`, so `check_cuda()` never runs and PyTorch raises on a CPU-only machine | drop the `or` and call `check_cuda()` when the flag is absent, or make the default `None` |
| B20 | `inputs==0.5` in `requirements.txt` | **unused** | imported by nothing in the tree | remove from the pin set |

## Part III — Prioritised open work

**P0 — without these, no number here is citable.**
1. Implement a demonstration split and held-out metrics (L1).
2. Compute and log the per-bit marginal baseline; report margin over it (L2).
3. Seed torch and run ≥5 seeds per architecture; publish mean ± SD (L3).
4. Add a `total_frames` + corpus-checksum parameter to every run so the dataset is identified (L7, L9).

**P1 — makes the existing claims meaningful.**
5. Re-run the benchmark across a budget sweep (10 / 30 / 100 / 300 epochs) (L4).
6. Reduce or exercise the dead action bits, and handle the idle-class imbalance (L5, L6).
7. Either re-record under the generic configuration or rename the packages so the corpus matches the
   claim (L8).
8. Make capture, demo-loading and width-reconciliation failures loud and counted (L9).

**P2 — correctness and hygiene fixes that are cheap and independent of the research.**
9. Fix the `train_imiation.py` glob and spelling (B4); `get_last_index` numeric ordering (B10).
10. Add logging to the deployment loops (B5).
11. Make the Windows-only imports conditional and repair the `Dockerfile` (B6).
12. Delete dead code and unwired config; derive MLflow names (B9, B11, B15, B16).
13. Drive gamepad emission from `actions.mappings` and resolve the `vg_code` mismatch (B12).
14. Remove hard-coded local paths (B13).

**P3 — structural work that pays off once the science is sound.**
15. Collapse `hajime_agent/` and `generic_agent/` into one package plus a per-title configuration
    (B14).
16. Make the recurrent encoder stateless, or episode-aware (L12).
17. Introduce a reward/terminal signal — a HUD or referee reader — so that episodic and closed-loop
    metrics become possible at all. Until then, "success rate" cannot be computed by this code base.
18. Version the corpus and checkpoints (Git LFS or release assets) so the study is externally
    reproducible.
19. Add tests and CI (B17); choose a licence (B18).

## Part IV — What can be stated honestly today

* A black-box, pixel-only screen-capture environment with virtual gamepad and keyboard/mouse actuation
  works well enough to record ~33 k frames of human demonstration and to train on them offline in
  dummy mode.
* The full BC pipeline (record → train → track → deploy) runs end to end on Windows with CUDA.
* Among six encoders, SB3's default NatureCNN is *clearly* the worst in-sample (≈0.5 nats worse than
  every custom encoder, far outside run-to-run spread).
* Replacing the flatten head with global average pooling cuts checkpoint size and parameter count by
  ~94 % while improving recorded loss and training time — a real, reproducible efficiency result
  (1.01 M parameters, 4.20 MB, 2.93 nats), independent of whether the loss is competitive with a
  baseline.
* ViT is parameter-cheap (2.45 M) and compute-expensive (55 min vs 18 s), which is a useful,
  quantified cost observation.
* Nothing about *playing ability* can yet be claimed.
