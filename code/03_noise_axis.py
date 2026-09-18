"""
step2d_corrected.py -- step 2 redone after its control decoder was shown to be broken.

WHAT WAS WRONG (verified, not conceded on authority)
-----------------------------------------------------
step2c_analyze.py had two independent faults, either of which alone invalidates
the headline:

1. NOISE MODEL. Noise std was scaled to each unit's OWN std (sig/snr), so every
   channel's variance was multiplied by the same (1 + 1/snr^2). Measured log-var
   shift vs predicted log(1+1/snr^2): +0.0615/+0.0606, +0.6913/+0.6931,
   +2.8263/+2.8332. A constant shift of all log-variance features; LDA is
   invariant to it. The manipulation could not move a log-variance decoder.

2. THE FEATURE WAS DESTROYED. Features were taken from data already z-scored
   along time, so var(axis=2) == 1 identically. Measured range
   [0.999999999983, 0.999999999998]; feature spread 1.5e-11 versus 4.60 for the
   same data un-z-scored. kappa_logvar = 1.000 was LDA separating floating-point
   round-off, not information. Pure structureless noise through that same
   pipeline scores kappa = -0.133 (nonzero, seed-dependent) -- the tell.

FIXES HERE
----------
* Noise has a COMMON ABSOLUTE scale across channels (global RMS / snr), so it
  changes between-channel variance RATIOS the way external sensor noise does.
* log-variance features are computed on UN-z-scored activity.
* The covariance path uses OAS only; z-scoring is applied to neither path, so
  both decoders see the same signal.
* A guard that actually fires: if the feature matrix has no spread, or if the
  same pipeline scores above chance on structureless input, the run FAILS.
"""
import glob, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score

CLASSES, N_PER, TOP_K = [1, 2, 3, 4], 15, 12
SNRS = [np.inf, 8.0, 4.0, 2.0, 1.0, 0.5, 0.25]
COND_MAX, MIN_SPREAD = 1e8, 1e-6


def separability_ratio(covs, labels):
    class_means = {}
    within = []
    for c in CLASSES:
        Cc = covs[labels == c]
        mc = mean_riemann(Cc)              # Riemannian (geometric) mean of the class
        class_means[c] = mc
        d = [distance_riemann(Cc[i], mc) for i in range(len(Cc))]
        within.append(np.mean(d))
    within_spread = np.mean(within)
    between = []
    for i in range(len(CLASSES)):
        for j in range(i + 1, len(CLASSES)):
            between.append(distance_riemann(class_means[CLASSES[i]],
                                            class_means[CLASSES[j]]))
    between_spread = np.mean(between)
    return between_spread / within_spread, between_spread, within_spread


def kap(pipe, X, y, seed=0):
    return cohen_kappa_score(y, cross_val_predict(
        pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed)))


def logvar_feats(X):
    return np.log(X.var(2) + 1e-30)


def sanity_or_die(A, y):
    """The check that step 2 needed and did not have."""
    f = logvar_feats(A)
    spread = float(f.max() - f.min())
    print(f"[guard] log-variance feature spread = {spread:.3e}", flush=True)
    if spread < MIN_SPREAD:
        sys.exit(f"!! feature spread {spread:.3e} < {MIN_SPREAD} -- the feature carries "
                 f"no information and any kappa from it is round-off. Aborting.")
    rng = np.random.default_rng(0)
    kn = [kap(make_pipeline(LDA()), logvar_feats(rng.normal(0, 1, A.shape)), y, s)
          for s in range(5)]
    print(f"[guard] kappa on structureless noise (5 seeds): "
          f"{np.round(kn, 3).tolist()}  mean {np.mean(kn):+.3f}", flush=True)
    if abs(np.mean(kn)) > 0.15:
        sys.exit("!! the decoder scores off-chance on structureless input. Aborting.")


def main():
    files = sorted(glob.glob("results/act/act_f*.npy"))
    if not files:
        sys.exit("no saved activity")
    y = np.repeat(CLASSES, N_PER)
    sanity_or_die(np.load(files[0]).astype(np.float64), y)

    rows = []
    for f in files:
        m = re.search(r"act_f([\d.]+)_s(-?\d+)\.npy", os.path.basename(f))
        frac, seed = float(m.group(1)), int(m.group(2))
        A = np.load(f).astype(np.float64)
        rms = A.std()                               # ONE scale for all channels
        for snr in SNRS:
            rng = np.random.default_rng(7)
            X = A if np.isinf(snr) else A + rng.normal(0, rms/snr, A.shape)

            raw = np.array([np.cov(x) for x in X])
            evs = np.linalg.eigvalsh(raw)
            ranks = np.array([np.linalg.matrix_rank(c) for c in raw])
            conds = np.array([e.max()/e.min() if e.min() > 0 else np.inf for e in evs])
            valid = bool(ranks.min() == TOP_K and conds.max() < COND_MAX)

            covs = Covariances(estimator='oas').transform(X)
            sep, betw, wit = separability_ratio(covs, y)
            k_lv = kap(make_pipeline(LDA()), logvar_feats(X), y)
            k_ts = kap(make_pipeline(TangentSpace(metric='riemann'), LDA()), covs, y)
            rows.append(dict(frac=frac, seed=seed, snr=snr, sep_ratio=sep,
                             kappa_logvar=k_lv, kappa_tangent=k_ts,
                             feat_spread=float(logvar_feats(X).max()-logvar_feats(X).min()),
                             min_rank=int(ranks.min()), cond_max=float(conds.max()),
                             valid=valid))
            print(f"  f={frac:.1f} s={seed:2d} snr={snr:>5} sep={sep:7.3f} "
                  f"k_lv={k_lv:+.3f} k_tan={k_ts:+.3f} {'OK' if valid else 'INVALID'}",
                  flush=True)
    df = pd.DataFrame(rows); df.to_csv("results/step2d_corrected.csv", index=False)

    d = df[df.valid]
    print("\n" + "="*74)
    print("corrected result")
    print("="*74)
    print(f"valid cells {len(d)}/{len(df)}")
    for k in ["kappa_logvar", "kappa_tangent"]:
        n_ceiling = int((d[k] >= 0.999).sum())
        print(f"  {k}: range [{d[k].min():+.3f}, {d[k].max():+.3f}], "
              f"{n_ceiling}/{len(d)} at ceiling, {d[k].round(3).nunique()} distinct")
        if d[k].nunique() >= 3:
            r, p = stats.pearsonr(d.sep_ratio, d[k])
            rs, ps = stats.spearmanr(d.sep_ratio, d[k])
            print(f"     vs sep_ratio: pearson r={r:+.3f} (p={p:.2g}) "
                  f"spearman rho={rs:+.3f} (p={ps:.2g})")
    print("\n--- intact network, SNR sweep ---")
    i = d[d.frac == 0.0].sort_values("snr", ascending=False)
    print(i[["snr", "sep_ratio", "kappa_logvar", "kappa_tangent"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
