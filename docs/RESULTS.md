# Results

> Every number in this document was re-read from the local MLflow file store
> (`generic_agent/mlruns/`, `hajime_agent/mlruns/`) and from the demo files themselves, not copied
> from a README. The store is **not version controlled** ([README §15.2](../README.md#152-what-is-and-is-not-versioned)),
> so these figures are currently reproducible only on the workstation that produced them.

## 1. Store census

| Experiment | ID | Runs | Status breakdown |
|---|---|---|---|
| `Model_Comparison` | `160445798900299176` | 26 | 13 finished · 8 abandoned mid-run (`RUNNING` in metadata, process long dead) · 4 failed at start · 1 completed summary |
| `Hajime_no_Ippo_Imitation_Learning` | `918368899121358984` | 6 | 4 finished · 2 failed at start |
| `Hajime_no_Ippo_Imitation_Learning` (hajime copy) | `923077394163286863` | 1 | finished; numerically identical to a run in the generic store |
| `Default` | `0` | 0 | created by an accidental `mlflow` call with no experiment set |

32 runs total. No run was ever deleted, and no run has an associated evaluation metric — the store
records optimisation telemetry only.

## 2. Completed benchmark runs (`Model_Comparison`)

All completed runs share the protocol of [README §9](../README.md#9-experimental-protocol): 10 epochs,
batch 384, lr 1e-4, `cuda`, seed 42, the same `generic_agent/notebooks/demos/` corpus.

| # | Encoder | Status | Final loss | Train time (s) | Wall clock (s) | Params | Checkpoint (MB) | `prob_true_act` | Logged `bc/loss` points |
|---|---|---|---|---|---|---|---|---|---|
| 1 | NatureCNN | finished | **3.476316** | 18.284 | 20.5 | 4,196,810 | 16.328 | 0.037193 | 2 |
| 2 | NatureCNN | finished | **3.581713** | 20.516 | 31.1 | 4,196,810 | 16.328 | 0.034255 | 2 |
| 3 | CNN+LSTM+Attn | finished | 3.024518 | 67.768 | 69.1 | 6,116,779 | 23.705 | 0.068065 | 2 |
| 4 | CNN+LSTM+Attn | failed (1.1 s) | — | — | — | — | — | — | 0 |
| 5 | ViT | finished | 2.998286 | 3,288.800 | 3,293.5 | 2,448,010 | 9.699 | 0.067101 | 2 |
| 6 | Impoola-CNN (GAP) | finished | 2.933126 | 226.170 | 243.3 | 1,009,258 | 4.197 | 0.069618 | 2 |
| 7 | Impala-CNN (Flatten) | finished | 2.945390 | 406.584 | 410.4 | 17,720,938 | 67.947 | 0.070320 | 2 |
| 8 | Impala-CNN (Flatten) | finished | 3.001048 | 351.735 | 355.8 | 17,720,938 | 67.947 | 0.067272 | 2 |
| 9 | Impala-CNN (Flatten) | finished | 3.027630 | 345.104 | 348.2 | 17,720,938 | 67.947 | 0.067582 | 2 |
| 10 | Impala-CNN (Flatten) | finished | 3.029788 | 535.953 | 544.5 | 17,720,938 | 67.947 | 0.067854 | 2 |
| 11 | ResNet-18 | finished | **2.901681** | 251.252 | 254.7 | 11,516,938 | 44.374 | 0.073433 | 2 |
| 12 | ResNet-18 | finished | 2.963942 | 442.330 | 444.9 | 11,516,938 | 44.374 | 0.068224 | 2 |
| 13 | ResNet-18 | finished | 3.018901 | 391.082 | 420.1 | 11,516,938 | 44.374 | 0.068146 | 2 |
| 14 | ResNet-18 | **abandoned** | — | — | — | — | — | 0.074819 | 2 (last `bc/loss` **2.851459**) |

Notes:

- `final_loss`, `training_time_s`, `num_params` and `model_size_mb` are logged explicitly at the end
  of `train_model()`; they are the authoritative per-run summary.
- `prob_true_act` is the joint sigmoid-bernoulli probability of the human's exact action vector at the
  last logged batch. Values are in [0,1] and higher is better.
- Every completed run logged exactly **two** `bc/loss` points across 10 epochs. Loss-logging density is
  set by `imitation`'s default `log_interval`, and at 27 k frames × 10 epochs / 384 ≈ 700 gradient steps
  the effective sampling of the curve is far too sparse to inspect convergence — the "curve" is two dots.
- Run 14 is the most informative abandoned record: a ResNet-18 that reached **2.851** before being
  interrupted, i.e. lower than any *completed* run. The 10-epoch budget, not the encoder, is what caps
  the reported numbers.

## 3. Headline table = one selected run per encoder

The published comparison ([README §10.1](../README.md#101-headline-benchmark),
`generic_agent/notebooks/models/comparison_results.md`) reports rows 1, 3, 5, 6, 9 and 11. That
selection is what `compare_models.py` produces in a single full pass: it trains each architecture once
and the last-written run per architecture is the one that reaches the generated table. The fallback
baselines baked into `compare_models.py` now carry these same six values; they previously carried
runs 7 and 13 (Impala 2.945390 / 406.6 s, ResNet 3.018901 / 391.1 s), which is how the README and the
generated report disagreed with each other before this revision.

## 4. Run-to-run dispersion

| Encoder | Completed runs | Loss min – max | Range | CV of time | `prob_true_act` range |
|---|---|---|---|---|---|
| NatureCNN | 2 | 3.4763 – 3.5817 | 0.1054 | 1.18× (18.3 → 20.5 s) | 0.0343 – 0.0372 |
| CNN+LSTM+Attn | 1 | — | — | — | 0.0681 |
| ViT | 1 | — | — | — | 0.0671 |
| Impoola-CNN | 1 | — | — | — | 0.0696 |
| Impala-CNN | 4 | 2.9454 – 3.0298 | 0.0844 | 1.31× (345 → 536 s) | 0.0673 – 0.0703 |
| ResNet-18 | 3 | 2.9017 – 3.0189 | 0.1172 | 1.39× (251 → 442 s) | 0.0681 – 0.0734 |

Two things this table establishes, and two it does not.

**Establishes:** NatureCNN's deficit (≈0.5 loss) is ~4× the largest intra-architecture spread, so
"the default encoder is worse than every custom encoder" is a supported claim. Parameter count,
checkpoint size and approximate training cost are deterministic and differ by more than an order of
magnitude across architectures.

**Does not establish:** any ordering among the four custom encoders. The published gaps are
2.90 < 2.93 < 3.00 < 3.02 < 3.03, a total span of 0.13, while a single architecture re-run moves by up
to 0.117. With `n = 1` for three architectures and no seed control (`compare_models.py` seeds only the
NumPy RNG; there is no `torch.manual_seed`, no `cudnn.deterministic`, and CUDA reductions are
non-deterministic), the ranking of rows 3–13 is within noise. A proper comparison needs ≥5 seeds per
architecture and a paired report (mean ± SD, plus an effect size).

## 5. Trivial baselines: what the reported losses actually mean

The BC objective in use is the sigmoid-bernoulli negative log-likelihood **summed over the
`num_actions` independent bits**, measured in nats (`imitation`'s `SigmoidBonus`-free default; the
action space is `MultiBinary(9)` for every row of the headline table). For a product-of-sigmoids
parameterisation, the in-sample optimum over the label distribution is attained when each bit's logit
reproduces that bit's empirical marginal, and its value is

```
L_marginal = Σ_i H(m_i)  where  m_i = mean over frames of action bit i
```

Computed directly from the reconciled 9-bit demonstration labels
(`generic_agent/notebooks/demos/`, 27,161 frames):

| Predictor | In-sample NLL (nats) | Derivation |
|---|---|---|
| Uniform (all logits 0) | **6.2383** | `9 · ln 2` |
| Per-bit marginal (best achievable without conditioning on the frame) | **2.6066** | `Σ H(m_i) = 0.6931 + 0.6614 + 0.3039 + 0.5099 + 0.3758 + 0.0378 + 0.0247 + 0 + 0` |
| **Best benchmark model (ResNet-18)** | **2.9017** | measured |
| Worst benchmark model (NatureCNN) | 3.4763 | measured |

The uniform value is a useful sanity check and it matches the tracking store exactly: every abandoned
`Swin` / `ConvNeXt` / `Vision_Mamba` run logs its first `bc/loss` between **6.226 and 6.241**, i.e.
those runs stopped at the untrained initialisation.

Two consequences, and they are the most important quantitative statements in this repository:

1. **Every architecture in the headline table is worse than the marginal predictor on its own training
   data.** The whole reported range, 2.90 – 3.48, sits above the 2.607 nats that a frame-independent
   per-bit marginal already achieves. The margin is +0.30 nats for the best model and +0.87 for the
   worst. At a 10-epoch budget the study therefore demonstrates **no conditional structure whatsoever**
   beyond the action prior; the architectures are being ranked on *how quickly they approach the
   marginal*, not on policy quality.
2. **This is a budget artefact, not an architectural verdict.** Run A of §6 below (300 epochs, same
   corpus, loss 1.330) is 1.28 nats *below* the marginal bound, at which point genuine conditioning has
   occurred. The benchmark simply stops before the regime where architecture differences become
   visible.

Recomputed per-bit marginals and entropies for both packages, at both reconciliation widths, are in
[DATA.md §4](DATA.md#4-action-distribution).

## 6. The budget is the binding constraint

Experiment `Hajime_no_Ippo_Imitation_Learning` contains the longer BC runs, using the same
architecture family but 100–300 epochs instead of 10.

| Run | Epochs | Last `bc/loss` | `prob_true_act` | Logged points | Wall clock | Status |
|---|---|---|---|---|---|---|
| A | 300 | **1.330473** | **0.482542** | 42 | 836.0 s | finished |
| B | 100 | 2.004686 | 0.285095 | 14 | 253.7 s | finished |
| C | 100 | 2.447901 | 0.114208 | 7 | 252.4 s | finished |
| D | 100 | 2.751739 | 0.075489 | 6 | 142.9 s | finished |
| E | 100 | — | — | 0 | 2.6 s | failed at start |
| F | 100 | — | — | 0 | 7.8 s | failed at start |

Run A is the source of the retained
[`progress.csv`](../generic_agent/notebooks/models/imitation/bc_logs/progress.csv) (43 rows, epoch
0 → 292, samples 384 → 7,872,384, loss 4.851 → 1.330, `prob_true_act` 0.0078 → 0.4826; `bc/l2_loss` is
`0.0` on every row because the gradient penalty term is disabled).

Three conclusions follow, and they apply to the whole study:

1. **Loss at epoch 10 is nowhere near convergence.** 10 epochs → ≈2.9–3.5; 300 epochs → 1.33. The
   benchmark compares architectures at a point where all of them are still descending steeply, which
   mostly measures early optimisation speed, not asymptotic capacity.
2. **Runs B, C and D are the same nominal configuration (100 epochs) and differ by 0.75 in loss**
   (2.00 vs 2.45 vs 2.75) and by a factor of 3.8 in `prob_true_act` (0.285 vs 0.114 vs 0.075). This is
   the clearest available evidence that uncontrolled variance dominates the architecture effect being
   studied. The runs are not, however, guaranteed to be identical in configuration — the scripts log
   `batch_size`, `learning_rate`, `epochs`, `device` and `model_type` but not the encoder
   hyperparameters, the demo set, or the torch/CUDA build, so they cannot be verified as replicates
   from the store alone.
3. **`prob_true_act` near 0.48 is still low** for a distribution over 24 observed joint actions
   ([README §6.2](../README.md#62-measured-corpus-statistics)). Even the longest run leaves most
   action vectors mis-assigned.

## 7. Abandoned architectures

| Encoder | Runs | Config in store | Last state | Source availability |
|---|---|---|---|---|
| `Swin_Transformer` | 1 | 2 epochs, batch 384 | `RUNNING`, one logged batch: loss 6.2363, `prob_true_act` 0.001945 | **deleted** |
| `ConvNeXt` | 2 | 10 epochs, batch 384 | one failed after 93.7 s with no metrics; one `RUNNING` at loss 6.2353, `prob_true_act` 0.001947 | **deleted** |
| `Vision_Mamba` | 6 | 1 epoch, batch 64 | four `RUNNING` at loss 6.226–6.241, `prob_true_act` ≈ 0.00195; two empty | **deleted**; only `__pycache__/mamba_architectures.cpython-311.pyc` remains |

Interpretation. All of them stall at `bc/loss` ≈ 6.23 with `prob_true_act` ≈ 0.00195 ≈ 2⁻⁹ — the
uniform baseline for a 9-bit `MultiBinary` space. That is the *first logged batch*, i.e. these runs
never got past initialisation, and their loss is exactly "the model has learned nothing". So there is
**no evidence at all** that these encoders are too slow for this task; there is evidence only that
six attempts were interrupted. The README's cost argument (≈28 M parameters, VRAM thrashing, ≈40 h for
10 epochs) is an engineering judgement the operator recorded in prose, and it is not supported by any
retained measurement — no peak-memory figure, no steps/second, no extrapolation table. If the claim
matters, it must be re-measured (§7).

`Vision_Mamba` is additionally unreproducible: its module was deleted and only the compiled `.pyc`
survives, so even the architecture cannot be re-instantiated from source.

## 8. Evidence gaps to close, in priority order

1. **Held-out evaluation.** A train/validation split over demonstrations, per-bit accuracy, per-bit
   F1, macro-averaged joint-action accuracy, and a top-k marginal baseline (predict the empirical
   action marginals) to separate "learned a conditional policy" from "learned the label prior". This
   is the single change that would make §2 mean anything.
2. **Closed-loop evaluation.** Requires an episodic signal the environment does not currently provide
   ([README §5.3](../README.md#53-reward-termination-and-the-deployment-loop)). Options: a
   round/life/HUD reader over the captured frame, or a scripted referee. Until one exists, no
   success-rate number can be produced.
3. **Replication.** ≥5 seeds per architecture, variance reported, and the ranking tested rather than
   asserted. Given §4 and §5(2), the current ranking is a hypothesis.
4. **Longer budget.** The 10-epoch table sits on the steepest part of the curve; a 100-epoch sweep
   would change which encoder looks best, since the encoders differ mainly in convergence rate.
5. **Retracted claims re-measured.** Peak VRAM and throughput for Swin/ConvNeXt, and a re-implementation
   of `Vision_Mamba` from the surviving bytecode's interface.
6. **DAgger and GAIL have zero runs.** Neither appears in the store
   ([README §8.2, §8.3](../README.md#8-training-objectives-and-algorithms)); both are unvalidated.
7. **Hajime package has no benchmark.** Its store holds a single 100-epoch BC run whose numbers are
   identical to a run in the generic store, and its 6,151-frame demonstration was never used to
   produce a comparison row.
