"""
step2_lesion.py -- does the classifier-free separability index track REAL
decodability when you damage a network whose wiring you know?

DESIGN
------
Pretrained flyvis (flow/0000/000). 4 classes = drifting-grating directions.
Lesion = zero a random fraction of the 604 (source_type -> target_type)
syn_strength parameters, i.e. delete whole pathways.

Controls that make this interpretable:
  * The SAME 60 stimulus movies are reused at every lesion level, so the only
    thing that changes between conditions is the connectome.
  * The 12 readout units are chosen ONCE at lesion=0 and then held FIXED, so
    the readout is not silently re-optimised per condition.
  * Two decoders are scored. TangentSpace+LDA shares the Riemannian geometry
    with the index, so on its own it would be a circular comparison. logvar+LDA
    does not -- it is the geometry-independent check, and it is the one to
    believe.
  * rank / condition number are recorded per condition. A condition whose
    covariance is singular produces a sep_ratio that means nothing, and is
    marked invalid rather than plotted.

Writes results/step2_lesion.csv incrementally.
"""
import os, sys, time
import numpy as np
import torch

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score

import flyvis
from flyvis import NetworkView

CLASSES   = [1, 2, 3, 4]
ANGLES    = {1: 0.0, 2: np.pi/4, 3: np.pi/2, 4: 3*np.pi/4}
N_PER_CLS = 15
N_FRAMES  = 300
DT        = 1/100.
SPAT_FREQ, TEMP_FREQ, NOISE = 0.18, 6.0, 0.05
TOP_K     = 12
BATCH     = 5
MODEL     = "flow/0000/000"
LEVELS    = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
LESION_SEEDS = [0, 1]
COND_MAX  = 1e8
OUT       = "results/step2_lesion.csv"


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


def build_stimuli(xy, rng):
    x, y = xy
    movies, labels = [], []
    t = np.arange(N_FRAMES) * DT
    for c in CLASSES:
        proj = x*np.cos(ANGLES[c]) + y*np.sin(ANGLES[c])
        for _ in range(N_PER_CLS):
            ph = 2*np.pi*(SPAT_FREQ*proj[None, :] - TEMP_FREQ*t[:, None]) + rng.uniform(0, 2*np.pi)
            mov = 0.5 + 0.5*np.sin(ph) + rng.normal(0, NOISE, (N_FRAMES, len(proj)))
            movies.append(np.clip(mov, 0, 1).astype(np.float32))
            labels.append(c)
    return np.asarray(movies), np.asarray(labels)


def simulate_all(net, movies, central):
    out = []
    for i in range(0, len(movies), BATCH):
        chunk = torch.from_numpy(movies[i:i+BATCH])[:, :, None, :]
        with torch.no_grad():
            act = net.simulate(chunk, dt=DT)
        out.append(act[:, :, central].numpy().transpose(0, 2, 1))   # (b, units, time)
        del act
    return np.concatenate(out).astype(np.float64)


def kappa_cv(featurizer, X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(featurizer, X, y, cv=cv)
    return cohen_kappa_score(y, pred)


def main():
    os.makedirs("results", exist_ok=True)
    nv = NetworkView(str(flyvis.results_dir / MODEL))
    net = nv.init_network(); net.eval()
    orig = net.edges_syn_strength.data.clone()
    n_conn = len(orig)

    df = net.connectome.nodes.to_df()
    r1 = df[(df.role == 'input') & (df.type == 'R1')].sort_values('index')
    uv = r1[['u', 'v']].to_numpy()
    u, v = uv[:, 0].astype(float), uv[:, 1].astype(float)
    xy = (u + v/2.0, v*np.sqrt(3)/2.0)
    central = np.array(net.connectome.central_cells_index)
    types = df.set_index('index').loc[central, 'type'].to_numpy()

    movies, y = build_stimuli(xy, np.random.default_rng(0))
    print(f"[stim] {len(y)} movies, reused at every lesion level", flush=True)

    # ---- intact run first: it fixes the readout for every later condition ----
    A0 = simulate_all(net, movies, central)
    keep = np.argsort(-A0.var(2).mean(0))[:TOP_K]
    print(f"[readout] fixed top-{TOP_K} units: {list(types[keep])}", flush=True)

    rows = []
    runs = [(0.0, -1)] + [(l, s) for l in LEVELS if l > 0 for s in LESION_SEEDS]
    for frac, seed in runs:
        t0 = time.time()
        if frac == 0.0:
            A = A0
            n_cut = 0
        else:
            rng = np.random.default_rng(1000 + seed)
            cut = rng.choice(n_conn, size=int(frac*n_conn), replace=False)
            net.edges_syn_strength.data.copy_(orig)
            net.edges_syn_strength.data[cut] = 0.0
            n_cut = len(cut)
            A = simulate_all(net, movies, central)
            net.edges_syn_strength.data.copy_(orig)

        X = A[:, keep, :]
        X = (X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-12)

        raw = np.array([np.cov(x) for x in X])
        evs = np.linalg.eigvalsh(raw)
        ranks = np.array([np.linalg.matrix_rank(c) for c in raw])
        conds = np.array([e.max()/e.min() if e.min() > 0 else np.inf for e in evs])
        valid = bool(ranks.min() == TOP_K and conds.max() < COND_MAX)

        covs = Covariances(estimator='oas').transform(X)
        sep, betw, wit = separability_ratio(covs, y)

        k_ts = kappa_cv(make_pipeline(TangentSpace(metric='riemann'), LDA()), covs, y)
        logvar = np.log(X.var(2) + 1e-12)
        k_lv = kappa_cv(make_pipeline(LDA()), logvar, y)

        row = dict(frac=frac, seed=seed, n_cut=n_cut, sep_ratio=sep, between=betw,
                   within=wit, kappa_tangent=k_ts, kappa_logvar=k_lv,
                   min_rank=int(ranks.min()), cond_max=float(conds.max()),
                   valid=valid, secs=round(time.time()-t0, 1))
        rows.append(row)
        import pandas as pd
        pd.DataFrame(rows).to_csv(OUT, index=False)
        print(f"  frac={frac:.1f} seed={seed:2d} cut={n_cut:3d}/{n_conn}  "
              f"sep={sep:6.3f}  k_tan={k_ts:+.3f}  k_logvar={k_lv:+.3f}  "
              f"rank={ranks.min()}/{TOP_K} cond={conds.max():.1e} "
              f"{'OK' if valid else 'INVALID'}  [{row['secs']}s]", flush=True)

    print("\nwrote", OUT, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
