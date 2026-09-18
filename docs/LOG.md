# Method log

The chronological record, including the parts that were wrong. Numbers here were
all measured in the runs this repository contains; `results/*.csv` has every
condition with validity flags.

---

## 1. Feasibility: can the data even feed the measure?

`code/01_feasibility.py` → `results/` diagnostics, printed

The index needs, per trial, a covariance matrix that is genuinely positive
definite. Three findings decided the whole design:

**API.** `Network.simulate(movie, dt)` returns `(batch, frames, 45669)`. Input
must be `(batch, frames, 1, hexals)` with `hexals = 721`, derived because
`n_hexals` is not a readable attribute: 5768 input-role neurons ÷ 8 photoreceptor
types = 721, which equals the extent-15 hex lattice 3·15²+3·15+1.
`connectome.central_cells_index` gives 65 indices, one central cell per type.

**An untrained network is degenerate.** With randomly initialised weights, **33 of
65 central cells have exactly zero variance** (relu + random init → dead units);
median channel variance is 0. Raw covariance rank 27-29 of 65, effective
dimension 11-19. The pretrained network has 0 dead units and effective dimension
63. Everything after this uses pretrained weights.

**A false PASS, and the fix.** The first SPD test was `min_eigenvalue > 0`. It
printed PASS on a rank-27-of-65 covariance, because the smallest eigenvalue was
`+1e-17`, numerically zero but positive. Shrinkage then returned finite numbers
and nothing crashed. The test was replaced with a **condition-number** criterion.

**Readout size.** Even pretrained, the response is intrinsically low-dimensional.
Condition number against number of readout channels (z-scored, 300 frames):

| k | 12 | 16 | 24 | 32 | 65 |
|---|---|---|---|---|---|
| cond | **1.2e5** | 2.6e8 | 6.7e13 | 1.7e14 | 1.2e15 |

12 channels is the only clearly safe setting. All later work uses the 12 most
variable central cells, re-derived per model.

---

## 2. A retracted result

`code/02*_retracted_*.py` · `figures/00_retracted_result.png`

This step reported an 11× dissociation: the separability index collapsing while a
"geometry-independent" decoder held κ = 1.000. **It was an artifact.** Two
independent bugs, either fatal alone:

**The noise model could not move the decoder.** Noise std was scaled to each unit's
own std, multiplying every channel's variance by the same `(1 + 1/snr²)`, a
constant shift of all log-variance features, and LDA is invariant to a constant
shift. Measured log-variance shift against the predicted `log(1+1/snr²)`:
+0.0615/+0.0606, +0.6913/+0.6931, +2.8263/+2.8332.

**The feature had been destroyed.** Features were taken from data already z-scored
along time, so `var(axis=2)` was identically 1. Measured range
`[0.999999999983, 0.999999999998]`; feature spread **1.5e-11**, against **4.60**
for the same data un-z-scored. The "perfect decoding" was LDA separating
floating-point round-off. Structureless Gaussian noise through the same pipeline
scores κ = −0.133: nonzero and seed-dependent, which is the tell.

**The guard that did not fire.** The script had a degeneracy check: fail if the
feature takes fewer than 3 distinct values. κ = 1.000 appeared in 68 of 77 cells
*alongside seven other values*, so `nunique` reached 8 and the check passed. **A
check that cannot fail in the case it exists for is not a check.** The replacement
runs the pipeline on structureless input and aborts if |κ| > 0.15.

The figure is kept so the retraction is legible rather than quietly deleted.

---

## 3. Noise axis, corrected

`code/03_noise_axis.py` · `figures/01_noise_axis.png` · `results/step2d_corrected.csv`

Fixes: noise given a **common absolute scale** across channels (global RMS / SNR),
so it changes between-channel variance *ratios* the way external sensor noise does;
log-variance features computed on **un-z-scored** activity; neither path z-scored,
so both decoders see the same signal; a guard that actually fires.

Intact network, SNR sweep:

| SNR | clean | 8 | 4 | 2 | 1 | 0.5 | 0.25 |
|---|---|---|---|---|---|---|---|
| index | 10.42 | 6.88 | 4.21 | 2.25 | 1.13 | 0.63 | 0.45 |
| κ geometry-independent | 1.000 | 1.000 | 1.000 | 0.978 | 0.889 | 0.356 | 0.156 |
| κ Riemannian | 1.000 | 1.000 | 1.000 | 1.000 | 0.711 | 0.022 | −0.044 |

Across 72 valid cells: geometry-independent ρ = **+0.926**; Riemannian ρ = +0.875.
The decoder that does *not* share the metric agrees at least as well, so the
agreement is not an artifact of measuring the same geometry twice.

A claim that briefly appeared here and was withdrawn: that the index is a
"leading, more sensitive indicator" because at SNR = 4 it had fallen to 40 % while
both decoders sat at 1.000. That rested on one cell of one row; across all 28
cells with SNR ≥ 4, κ spans 0.111-1.000. More basically, the index is an unbounded
ratio and κ is bounded above by 1, so comparing their percentage drops compares
units, not sensitivity.

---

## 4. Calibration: what makes the task harder, and what doesn't

`code/04_calibration.py` · `results/step3_calib.csv`, `results/step3_operating_point.csv`

**Class similarity does not work.** Shrinking the angular separation between the
four motion classes from 45° to 2° left κ = 1.000 at every step while the index
fell 10.42 → 1.20:

| separation | 45° | 20° | 12° | 7° | 4° | 2° |
|---|---|---|---|---|---|---|
| κ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| index | 10.42 | 7.40 | 5.17 | 3.45 | 2.19 | 1.20 |

The network is deterministic, so within-class scatter is near zero; classes 2°
apart are still perfectly separable. Difficulty here comes from within-class
variability, not between-class distance.

**Observation SNR does.** SNR = 1.0 puts the intact network at κ ≈ 0.77. That is
the operating point for everything after.

---

## 5. Damage axis

`code/05_damage_axis.py` · `figures/02_damage_axis.png` · `results/step4_fine.csv`

0-25 % of the 604 pathways deleted, 3 lesion seeds × 3 noise seeds, 48 rows,
**0 at ceiling**.

**A correction to how this was first analysed.** The original version treated all
48 rows as independent and reported a correlation within each lesion dose,
"significant at every dose". Those 48 rows are 16 lesions each measured under 3
observation-noise draws, so the three draws of one lesion are not independent
observations, and an n = 9 p-value inside a dose treats them as if they were. That
is pseudo-replication, and two visible symptoms followed from it: the per-dose
result held under Pearson but was 14/15 under Spearman, and only 9 of 15 survived
a Bonferroni correction the 15 tests needed and never got.

The unit of analysis is now **one lesion**, with its noise draws averaged: 15
independent lesions per model. For this model, *r* = **+0.939** (p = 2.2e-07).

Dose is then removed properly rather than by stratifying into cells of three.
Dose alone predicts κ weakly and not significantly (*r* = -0.339, p = 0.22), and
the partial correlation holding dose constant is **+0.934** (p = 1.1e-06, df = 12),
essentially identical to the raw value. The stronger claim survives the stricter
analysis.

Variation in κ across lesion *seeds* within a dose (SD 0.268) exceeds variation
across *doses* (SD 0.203): which pathways were cut matters more than how many.

---

## 6. Replication across the ensemble

`code/06_ensemble.py` · `figures/03_ensemble_replication.png` · `results/step5_all_models.csv`

Three independently trained networks, same protocol, readout re-derived per model.

| model | intact κ | r | ρ |
|---|---|---|---|
| 000 | 0.770 | +0.900 | +0.897 |
| 001 | 0.993 | +0.953 | +0.956 |
| 002 | 0.800 | +0.945 | +0.919 |
| pooled (n = 144) | n/a | **+0.934** | **+0.950** |

Those per-model values are computed on all 63 rows each and are therefore
pseudo-replicated in the same way as §5. On **independent lesions** (noise draws
averaged, 15 per model) the numbers are:

| model | n | r | p | partial r, dose removed | p |
|---|---|---|---|---|---|
| 000 | 15 | +0.939 | 2.2e-07 | **+0.934** | 1.1e-06 |
| 001 | 15 | +0.964 | 7.0e-09 | **+0.966** | 1.9e-08 |
| 002 | 15 | +0.962 | 1.1e-08 | **+0.957** | 8.0e-08 |

Fisher-z combined across the three models: *r* = **+0.956**, 95 % CI
[+0.918, +0.977]. Three tests rather than fifteen, each far past any correction
those three would need, and dose removed by partialling rather than by splitting
into cells of three.

A protocol finding worth recording: the SNR operating point does **not** transfer.
At the same SNR = 1.0, intact κ was 0.770 / 0.993 / 0.800; model 001 is markedly
more robust to the same absolute noise, pushing 8 of its 48 cells to ceiling.
Excluding them moves its correlation by 0.002, but a future run should calibrate
SNR per model rather than assume 1.0 transfers.

The top-12 readouts differ per model but share `Mi4`, `Mi1`, `Am`, `TmY18` and a
T4 subtype, the direction-selective and medulla-intrinsic types a motion task
should recruit.

---

## 7. Specificity: targeted vs matched-random lesions

`code/07_targeted.py` · `figures/04_specificity.png` · `results/step6_both_models.csv`

**Pathway indexing.** `edges_syn_strength` is grouped `as_index=False, sort=False`,
so its order is first-appearance in the edge table, not sorted order. Lesioning
the wrong pathways would produce clean-looking nonsense, so the ordering is
verified on every run: `edges_sign` is grouped identically and each pathway's sign
is a fixed property of the connectome, so the sign vector implied by our ordering
must equal the library's own `edges_sign` element for element: **604/604, zero
mismatches**, with a shuffled-order control that must fail (it does).

An earlier version verified this behaviourally: zero `R1 → L1`, check L1 moves
most. That passed on one model and **falsely failed** on another, because L1 is
driven by R1-R6 and removing one sixth of its drive is not always the largest
downstream change. A check that fires when nothing is wrong is as bad as one that
never fires; the structural test has neither failure mode and needs no simulation.

**Result.** Each anatomically targeted set against three random lesions of exactly
the same size:

| model | targeted | n | index | κ | random κ (mean) | κ below all 3 | index flags |
|---|---|---|---|---|---|---|---|
| 000 | T4 canonical input | 16 | 0.751 | 0.281 | 0.570 | no | no |
| 000 | → T5 | 35 | 0.886 | 0.607 | 0.662 | yes | yes |
| 000 | → T4 | 44 | 0.748 | 0.259 | 0.467 | no | no |
| 000 | ← photoreceptor | 44 | 0.477 | 0.030 | 0.467 | yes | yes |
| 000 | → T4+T5 | 79 | 0.470 | 0.148 | 0.556 | yes | yes |
| 001 | T4 canonical input | 16 | 0.819 | 0.474 | 0.965 | yes | yes |
| 001 | → T5 | 35 | 1.760 | 0.993 | 0.919 | no | no |
| 001 | → T4 | 44 | 0.730 | 0.385 | 0.812 | yes | yes |
| 001 | ← photoreceptor | 44 | 0.478 | 0.030 | 0.812 | yes | yes |
| 001 | → T4+T5 | 79 | 0.737 | 0.363 | 0.825 | yes | yes |

Seven hurt; the index flagged 7/7. Three did not; the index flagged 0/3, ten out
of ten.

**How much of that is a real test.** Two of the ten are sanity checks rather than
tests. Cutting all photoreceptor output in a visual task, and cutting all input to
both direction-selective families in a motion task, are destructive by
construction; a measure that missed them would be broken, and catching them is not
evidence of discrimination. Dropping those leaves six informative comparisons:
`onto T4`, `onto T5` and `T4 canonical input` in each of two models. The decoder
calls three of the six destructive and three not, and the index calls all six the
same way. Under an exact null where the index must choose which three of six to
flag, p = 1/C(6,3) = **0.05**.

A second deflation: the index and the decoder are already strongly correlated
inside each four-point comparison set (mean Spearman +0.881, and exactly +1.00 in
5 of the 10 sets), so agreement on "is the targeted point the lowest" is partly
entailed by §5 and §6 rather than independent of them. The reason the six-way
result is still worth stating is its balance: three positives and three negatives
means it rests on the index correctly saying **no**, which a pure sensitivity
result never shows.

Pooled across all 126 rows, *r* = +0.846, ρ = +0.921, with the same
pseudo-replication caveat as §5 and §6.

**What is not claimed.** Which pathways matter is model-dependent: `onto_T4` is
below all randoms in one network and inside the random range in the other. Two
networks trained on the same connectome distribute the task differently, and n = 2
supports no statement of the form "cutting X always hurts most". The claim is
about the index agreeing with the decoder case by case.

**A flaw in this run.** The random-control seed was `5000 + s + 17*len(idx)` -
dependent on seed and *size* only. Two targeted sets are both 44 pathways, so
their "separate" control sets are literally the same three draws. Valid as
controls, since the null for a given dose is the same, but they are one set of
three rather than two, and three conditions per model were wasted. A future run
must fold the target-set name into the seed.

---

## Compute

Roughly 60 s per condition on an Apple M2 (4 performance + 4 efficiency cores),
for 60 trials of 300 frames.

`Network.simulate` integrates 300 sequential timesteps over 1.51 M synapses and
cannot be parallelised across steps, so wall time scales with core speed rather
than core count. Apple MPS was detected but raised inside `simulate`; the scripts
fall back to CPU automatically.

Every script is resumable: a completed condition is written to its CSV
immediately and skipped on restart. This was not decoration: one long run was
killed twice by its host partway through, and the resume logic recovered both
times with nothing lost.
