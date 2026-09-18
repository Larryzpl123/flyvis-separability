"""
step2c_analyze.py -- the actual question, on a 2-D grid.

Two ways to break a recording, crossed:
  LESION  : delete connectome pathways        -> genuine neural deficit
  SNR     : add observation noise to the readout -> recording / method deficit

For every cell of the grid: the classifier-free separability index, and two
decoders. The claim under test is that the index TRACKS real decodability.
If it does not, say so -- a flat or non-monotonic relation is the result.

Reuses activity saved by step2b_grid.py, so no re-simulation.
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

CLASSES  = [1, 2, 3, 4]
N_PER    = 15
SNRS     = [np.inf, 4.0, 2.0, 1.0, 0.5, 0.25, 0.12]
COND_MAX = 1e8
TOP_K    = 12


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


def kappa_cv(pipe, X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cohen_kappa_score(y, cross_val_predict(pipe, X, y, cv=cv))


def main():
    files = sorted(glob.glob("results/act/act_f*.npy"))
    if not files:
        sys.exit("no saved activity; run step2b_grid.py first")
    y = np.repeat(CLASSES, N_PER)
    rows = []
    for f in files:
        m = re.search(r"act_f([\d.]+)_s(-?\d+)\.npy", os.path.basename(f))
        frac, seed = float(m.group(1)), int(m.group(2))
        A = np.load(f).astype(np.float64)              # (trials, units, time)
        sig = A.std(2, keepdims=True)                  # per-trial per-unit scale
        for snr in SNRS:
            rng = np.random.default_rng(7)
            X = A if np.isinf(snr) else A + rng.normal(0, 1, A.shape) * (sig / snr)
            X = (X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-12)

            raw = np.array([np.cov(x) for x in X])
            evs = np.linalg.eigvalsh(raw)
            ranks = np.array([np.linalg.matrix_rank(c) for c in raw])
            conds = np.array([e.max()/e.min() if e.min() > 0 else np.inf for e in evs])
            valid = bool(ranks.min() == TOP_K and conds.max() < COND_MAX)

            covs = Covariances(estimator='oas').transform(X)
            sep, betw, wit = separability_ratio(covs, y)
            k_ts = kappa_cv(make_pipeline(TangentSpace(metric='riemann'), LDA()), covs, y)
            k_lv = kappa_cv(make_pipeline(LDA()), np.log(X.var(2) + 1e-12), y)
            rows.append(dict(frac=frac, seed=seed, snr=snr, sep_ratio=sep,
                             between=betw, within=wit, kappa_tangent=k_ts,
                             kappa_logvar=k_lv, min_rank=int(ranks.min()),
                             cond_max=float(conds.max()), valid=valid))
            print(f"  f={frac:.1f} s={seed:2d} snr={snr:>5} sep={sep:7.3f} "
                  f"k_tan={k_ts:+.3f} k_lv={k_lv:+.3f} {'OK' if valid else 'INVALID'}",
                  flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/step2c_grid.csv", index=False)

    print("\n" + "="*74)
    print("does the classifier-free index track real decodability?")
    print("="*74)
    d = df[df.valid]
    print(f"valid cells: {len(d)}/{len(df)}")
    for kcol in ["kappa_logvar", "kappa_tangent"]:
        sub = d[[kcol, "sep_ratio"]].dropna()
        if sub[kcol].nunique() < 3:
            print(f"  {kcol}: DEGENERATE ({sub[kcol].nunique()} distinct values) -- "
                  f"no dynamic range, correlation meaningless")
            continue
        r, p = stats.pearsonr(sub["sep_ratio"], sub[kcol])
        rs, ps = stats.spearmanr(sub["sep_ratio"], sub[kcol])
        print(f"  {kcol}: pearson r={r:+.3f} (p={p:.2g})  spearman rho={rs:+.3f} (p={ps:.2g})  n={len(sub)}")

    print("\n--- mean over lesion seeds, by cell ---")
    piv = d.pivot_table(index="frac", columns="snr",
                        values=["sep_ratio", "kappa_logvar"], aggfunc="mean")
    print(piv.round(3).to_string())
    print("\nwrote results/step2c_grid.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
