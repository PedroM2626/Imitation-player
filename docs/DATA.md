# Data

> The demonstration corpus is the entire supervised signal in this project. This document specifies
> the recording procedure, the on-disk format, and the measured statistical properties of the corpus
> that produced every number in [RESULTS.md](RESULTS.md). All statistics were recomputed by reading
> the `.pt` files directly.

## 1. Where the data lives, and whether you can get it

| Path | Contents | Size | Versioned |
|---|---|---|---|
| `runs/hajime_ippo/demos/demo_*.pt` (four sessions recorded under the generic configuration) | 4 trajectories, 27,161 action frames | ~3.1 GB total | **no** — `*.pt` and `runs/*/demos/` are git-ignored |
| `runs/hajime_ippo/demos/demo_0_20260619_103226.pt` (the Hajime session) | 1 trajectory, 6,150 action frames | 546 MB | **no** |
| `runs/hajime_ippo/models/bc_policy.zip` | reference BC checkpoint | 17 MB | **no** — `*.zip` is git-ignored; the reference policy is exported as a release asset |
| `runs/hajime_ippo/logs/progress.csv` | 43-row training log of the 300-epoch run | 6 KB | yes |

Data lives outside the code, rooted at `runs/<profile>/`; `IMITATION_RUNS` or `--runs-root` relocate the
root. `progress.csv` and `models/comparison_results.md` are the only run artefacts under version control.

The corpus is therefore **single-machine data**. Nothing in the repository lets a reader obtain the
demonstrations that produced the benchmark; §7 states what would be required.

## 2. Recording protocol

`agent/cli/record.py` (run as `python -m agent.cli.record --profile <name>`) is the only writer of
demonstration files.

### 2.1 Loop

```
construct GenericGameEnv(cfg)                    # live capture, dummy = False
  wait for the game window (up to 120 s)
  create the virtual pad / input bridge
loop:
  read the physical input device                 # XInput via ctypes.windll.xinput1_4/xinput1_3,
                                                 #   keyboard/mouse via `keyboard` + `mouse`
  build the action vector from the live inputs
  env.step(action)                               # emit to the virtual device + grab the next frame
  append (frame, action) to the in-memory buffer
  handle hotkeys
```

Hotkeys, as implemented: `K` starts and stops recording, and **each stop is what writes a file**; `ESC`,
the window close button and Ctrl+C now flush an in-progress recording through
`TrajectoryRecorder.finish()` before exiting (previously only the `K` stop-transition wrote one, so an
interrupted take was lost), which makes the overlay text "`[ESC] Save & Exit`" accurate
([TRAINING_GUIDE §3.1](TRAINING_GUIDE.md#31-protocol)).
One recorder serves both profiles: `agent/utils/input_map.HumanInput` reads the key-mapping table in
`keyboard_mouse` mode and the physical pad (or its keyboard stand-ins) in gamepad mode, sized to the
profile's `actions.num_actions`.

### 2.2 Written artefact

```
runs/<profile>/demos/demo_<index>_<YYYYMMDD>_<HHMMSS>.pt
```

Contents: a Python **list containing one** `imitation.data.types.Trajectory`, serialised with
`torch.save`. The recorder constructs it as
`Trajectory(obs=…, acts=…, infos=None, terminal=False)`, so `rews` is absent from the stored objects —
sound for BC, but it means these files cannot be fed to any reward-consuming algorithm.

| Field | Shape / value | dtype | Meaning |
|---|---|---|---|
| `obs` | `(N+1, 4, 128, 128)` | `uint8` | grayscale frames, already **stacked 4-deep at capture time**, CHW. The first frame is appended twice, which is where the extra row comes from |
| `acts` | `(N, num_actions)` | `float32` | binary-valued action vector in `[0,1]`, one per transition |
| `infos` | `None` | — | never populated |
| `terminal` | `False` | `bool` | never populated |

Because `infos` and `terminal` are constants, a file records **one uninterrupted recording session**
with no episode structure inside it, and several `K` toggles produce several independent files rather
than several trajectories in one file.

The `N+1 / N` asymmetry is deliberate: the trailing observation is the outcome of the last action, so
`obs[t]` pairs with `acts[t]` and `obs[t+1]` is the next state. The loaders depend on this alignment;
a recorder that appended the frame *after* the action would silently shift every label by one frame.

Frame stacking at capture time is what allows `TemporalAttentionLSTM`-style code to find
`observation_space.shape[0] == 4`. It also means **the demos and the live environment are not the same
shape**: the env declares `(128, 128, 1)` and only the `VecFrameStack` wrapper makes it `(4, 128, 128)`.

## 3. Measured corpus inventory

Recomputed from the files on disk:

All five files now live in `runs/hajime_ippo/demos/`; the "Configuration" column names the recording
setup each session was captured under, not a separate data directory.

| Configuration | Session | Frames (`acts`) | Width | Distinct joint actions | Modal joint action share | Top-5 share | Joint entropy |
|---|---|---|---|---|---|---|---|
| generic | `demo_0_20260523_221206` | 6,301 | 18 | 35 | 0.2595 | 0.7670 | 2.158 bits |
| generic | `demo_1_20260523_221830` | 7,068 | 18 | 37 | 0.2473 | 0.7172 | 2.264 bits |
| generic | `demo_0_20260523_224241` | 8,149 | **7** | 36 | 0.2746 | 0.8165 | 2.059 bits |
| generic | `demo_1_20260523_224751` | 5,643 | **7** | 39 | 0.1845 | 0.6814 | 2.413 bits |
| generic | **pooled, reconciled to 18** | **27,161** | 18 | **45** | 0.2485 | 0.7261 | **2.286 bits** |
| hajime | `demo_0_20260619_103226` | 6,150 | 18 | **24** | 0.5127 | 0.7995 | 1.875 bits |

Wall-clock duration is not stored. At the configured `target_fps` of 60 the generic pool is ≈ 7.5
minutes of play and the hajime session ≈ 1.7 minutes; because capture is asynchronous and the recorder
may sample faster or slower than the emulator presents frames, treat these as upper bounds.

## 4. Action distribution

Pooled per-bit marginals after reconciliation to the declared width (the numbers that generate the
loss baseline in [RESULTS.md §5](RESULTS.md#5-trivial-baselines-what-the-reported-losses-actually-mean)):

| Bit | Semantic | Generic marginal | Generic per-bit entropy (nats) | Hajime marginal | Hajime entropy (nats) |
|---|---|---|---|---|---|
| 0 | up / `W` | 0.4953 | 0.6931 | 0.2655 | 0.5788 |
| 1 | down / `S` | 0.3748 | 0.6614 | 0.0868 | 0.2951 |
| 2 | left / `A` | 0.0906 | 0.3039 | 0.1953 | 0.4938 |
| 3 | right / `D` | 0.2070 | 0.5099 | 0.0654 | 0.2415 |
| 4 | A / Cross / `SPACE` | 0.1245 | 0.3758 | 0.0324 | 0.1428 |
| 5 | B / Circle / `F` | 0.0062 | 0.0378 | 0.0686 | 0.2500 |
| 6 | X / Square / `R` | 0.0038 | 0.0247 | 0.0200 | 0.0980 |
| 7 | left trigger | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 8 | right trigger | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 9–17 | stick press, camera x/y ×4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | **sum** | | **2.6066** | | **2.1001** |

Four structural properties:

1. **11 of 18 action bits never fire.** Triggers, stick press and the entire discretised camera axis
   (bits 10–17) are exactly zero in every file of both recording configurations. The declared action
   space is 18 dimensional; the *used* action space is 7 dimensional.
2. **Severe class imbalance.** Bits 0 and 1 account for ~87 % of all actuations in the generic pool;
   bits 5 and 6 fire below 1 %.
3. **The observed joint distribution is tiny and concentrated.** 45 distinct joint vectors occur in
   27,161 generic frames (of `2^18` = 262,144 possible), and the five most common cover 72.6 % of the
   data. In the hajime session only 24 distinct vectors occur, and the modal one is *the all-zero
   "do nothing" vector at 51.3 % of frames*.
4. **Total entropy is 2.29 bits** (generic) / **1.88 bits** (hajime) for a nominally 18-bit channel.
   The label distribution is nearly degenerate.

Consequences for modelling. A product-of-sigmoids policy can only represent independent per-bit
Bernoullis, so its in-sample optimum on this corpus is `Σ H(m_i)` = **2.6066 nats** (generic, 9-bit
reconciliation). Anything above that value indicates the model has not even matched the marginal;
anything below it proves it has begun to condition on the frame. Every row of the published benchmark
sits **above** 2.6066 ([RESULTS.md §5](RESULTS.md#5-trivial-baselines-what-the-reported-losses-actually-mean)).

Consequences for the *task*. Because camera and trigger bits never fire, nothing in this corpus can
teach a camera-control or blocking behaviour. Any claim that the trained agent "plays the game"
requires either demonstrations that exercise those bits or a task metric that does not depend on them.

## 5. Observation distribution

| Property | Generic pool | Hajime session |
|---|---|---|
| dtype / range | `uint8`, observed min 0, max 255 | same |
| Mean pixel intensity | 159.6 – 165.7 (per file) | in the same band |
| Resolution | 128 × 128, single channel ×4 stacked | same |
| Source resolution | 854 × 480 capture box, `INTER_NEAREST` downsample | same |

No normalisation, whitening or running observation statistics are applied in the environment; each
extractor normalises internally (`/255` then `(x − 0.5)/0.5` where implemented), and NatureCNN does not
normalise at all. The mean intensity near 160 (out of 255) means the frames are dominated by a bright
background — consistent with a fighting-game stage — so any encoder that keys on overall brightness has
an easy, and useless, shortcut.

## 6. Integrity problems, stated plainly

### 6.1 Mixed action widths inside one corpus
Two of the four generic files are 18-wide and two are 7-wide, while the `roblox` profile
(`agent/config/profiles/roblox.py`, the configuration the generic sessions were recorded under) declares
`num_actions = 9`. The published benchmark reconciled that by truncating `18 → 9` and zero-padding
`7 → 9` silently, with no message emitted, which is how a 9-bit label space was assembled from data
recorded under two different action regimes, for a game that is not the one that profile targets. That
reconciliation is no longer the default: `agent/utils/demos.load_demos()` runs under the profile's
`actions.width_policy`, whose default `strict` raises `DemoError` naming the file and both widths, and
the old behaviour is available only as an explicit, per-file-warned `coerce` (`--width-policy coerce`).
Because both source widths encode the same first seven semantics, the reconciled labels are *coherent* —
but re-running the published configuration on this corpus now refuses to load it unless the operator opts
in.

### 6.2 Failures used to be swallowed
Demo loading was wrapped per-file in `try/except Exception` that printed one line and continued, so a
corrupted, wrong-width or unreadable trajectory reduced the dataset without failing the run. That is now
a hard error: `load_demos()` collects every unreadable file and raises `DemoError` listing all of them
(`agent/utils/demos.py:92`). The corpus size is also recorded — `agent.utils.demos.summarise()` computes
the frame count, trajectory count, action width and distinct joint-action count, and
`agent.utils.tracking.log_dataset()` writes them, with the two reference losses, into every run, so a
benchmark trained on half the data is no longer indistinguishable from one trained on all of it. There is
still no assertion that the loaded count matches an *expected* count, and no split to check it against.

### 6.3 Capture can degrade invisibly
If a grab returns nothing, the previous frame is re-emitted (or a black frame when no previous frame
exists). The environment now increments `self.dropped_frames` rather than degrading wholly unremarked,
but the counter is neither printed nor carried into the demonstrations, so stored stale frames remain
indistinguishable from a static game screen. Since the label is the human's input at that instant, a
stale frame produces a `(frozen screen, new action)` pair that teaches the model the screen does not
matter.

### 6.4 No provenance metadata
Demonstration files carry no recording configuration: not the capture geometry, not the game build, not
the operator, not the `num_actions` in force at recording time, not the seed of the RNG used by the
recorder. The width discrepancy in §6.1 is only discoverable by opening the files. A sidecar JSON per
demo (or `infos` populated instead of left as empty dicts) would make the corpus self-describing.

### 6.5 No train/val/test partition exists
Every script uses the whole directory `runs/<profile>/demos/` as training data. There is no split, no
held-out demonstration, and no leave-one-session-out scheme anywhere in the repository. This is the root
cause of the absence of generalisation numbers, not a mere missing script.

## 7. What the corpus needs

Ordered by how much each changes what can be claimed:

1. **A held-out split.** Even a leave-one-file-out scheme over the four generic sessions gives a real
   generalisation number at zero recording cost.
2. **Demonstrations that exercise bits 7–17**, or an honest reduction of the action space to the 7
   bits that are used, with the loss re-derived so cross-study comparisons are possible. Those bits are
   no longer unreachable by code: `agent/utils/input_map.py` gives every entry of the gamepad table a
   physical source (triggers via `bLeftTrigger`/`bRightTrigger`, camera via `sThumbRX/RY`), so this is now
   purely a recording task.
3. **Volume.** 27 k frames at 128×128 is small by any standard for pixel-based imitation; §RESULTS 6
   shows the models had not converged by 10 epochs and had not beaten the marginal predictor either.
   The 300-epoch run reaching 1.33 nats indicates the existing data still rewards more optimisation
   before more recording is needed.
4. **Balanced classes.** The hajime session is half "do nothing". Either subsample idle frames or
   weight the loss; an unweighted product-of-sigmoids trained on 51 % no-op frames learns to idle.
5. **Versioned, self-describing data.** A checksummed, metadata-bearing export — Git LFS for small
   corpora, or a release asset / datasetDOI for large ones — so the study in README §10 is reproducible
   by someone other than the operator.
6. **A per-bit marginal baseline computed and logged inside the training scripts**, so that a run which
   fails to beat `Σ H(m_i)` cannot be reported as a result. This is now in place —
   `agent.utils.demos.summarise()` computes `marginal_baseline_nats` and
   `agent.utils.tracking.log_dataset()` logs it, and `agent.cli.train` prints the margin of the final
   loss against it — but nothing yet *refuses* to report a run above the baseline.
