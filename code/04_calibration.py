"""
step3_calib.py -- find the angular class separation that puts the
geometry-independent decoder near kappa = 0.8, so the lesion axis has room to move.

NOTE: an earlier draft of this file repeated the exact bug that invalidated
step 2 -- it z-scored along time and THEN took log(var(axis=2)), which is
identically 1. Features here are computed on UN-z-scored activity, and the same
structureless-input guard that step2d uses runs first.
"""
import numpy as np, torch, time, sys
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import cohen_kappa_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
import flyvis
from flyvis import NetworkView

CLASSES = [1, 2, 3, 4]
N_PER, N_FRAMES, DT = 15, 300, 1/100.
SPAT_FREQ, TEMP_FREQ, NOISE, TOP_K, BATCH = 0.18, 6.0, 0.05, 12, 5
DELTAS_DEG = [45, 20, 12, 7, 4, 2]
MIN_SPREAD, COND_MAX = 1e-6, 1e8

nv = NetworkView(str(flyvis.results_dir / "flow/0000/000"))
net = nv.init_network(); net.eval()
df = net.connectome.nodes.to_df()
r1 = df[(df.role == 'input') & (df.type == 'R1')].sort_values('index')
uv = r1[['u', 'v']].to_numpy(); u, v = uv[:, 0].astype(float), uv[:, 1].astype(float)
XC, YC = u + v/2.0, v*np.sqrt(3)/2.0
central = np.array(net.connectome.central_cells_index)
y = np.repeat(CLASSES, N_PER)


def movies_for(angles, rng):
    t = np.arange(N_FRAMES)*DT; out = []
    for ang in angles:
        proj = XC*np.cos(ang) + YC*np.sin(ang)
        for _ in range(N_PER):
            ph = 2*np.pi*(SPAT_FREQ*proj[None, :] - TEMP_FREQ*t[:, None]) + rng.uniform(0, 2*np.pi)
            m = 0.5 + 0.5*np.sin(ph) + rng.normal(0, NOISE, (N_FRAMES, len(proj)))
            out.append(np.clip(m, 0, 1).astype(np.float32))
    return np.asarray(out)


def sim(movs):
    o = []
    for i in range(0, len(movs), BATCH):
        with torch.no_grad():
            a = net.simulate(torch.from_numpy(movs[i:i+BATCH])[:, :, None, :], dt=DT)
        o.append(a[:, :, central].numpy().transpose(0, 2, 1)); del a
    return np.concatenate(o).astype(np.float64)


def logvar(X):
    return np.log(X.var(2) + 1e-30)


def kap(pipe, X, yy, seed=0):
    return cohen_kappa_score(yy, cross_val_predict(
        pipe, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=seed)))


def sep_of(covs, yy):
    cm = {c: mean_riemann(covs[yy == c]) for c in CLASSES}
    wi = np.mean([np.mean([distance_riemann(C, cm[c]) for C in covs[yy == c]]) for c in CLASSES])
    be = np.mean([distance_riemann(cm[i], cm[j]) for i in CLASSES for j in CLASSES if i < j])
    return be/wi


keep = None
print(f"{'delta':>7} {'k_logvar':>9} {'k_tan':>7} {'sep':>8} {'spread':>9} {'cond':>9} {'secs':>5}", flush=True)
rows = []
for dd in DELTAS_DEG:
    t0 = time.time()
    A = sim(movies_for([np.deg2rad(dd*i) for i in range(4)], np.random.default_rng(0)))
    if keep is None:
        keep = np.argsort(-A.var(2).mean(0))[:TOP_K]
        rng = np.random.default_rng(0)
        kn = [kap(make_pipeline(LDA()), logvar(rng.normal(0, 1, A[:, keep, :].shape)), y, s)
              for s in range(5)]
        print(f"[guard] structureless-input kappa: {np.round(kn,3).tolist()} mean {np.mean(kn):+.3f}",
              flush=True)
        if abs(np.mean(kn)) > 0.15:
            sys.exit("!! decoder scores off-chance on structureless input. Aborting.")
    X = A[:, keep, :]
    f = logvar(X); spread = float(f.max() - f.min())
    if spread < MIN_SPREAD:
        sys.exit(f"!! feature spread {spread:.2e} -- feature destroyed. Aborting.")
    covs = Covariances(estimator='oas').transform(X)
    evs = np.linalg.eigvalsh(np.array([np.cov(x) for x in X]))
    cond = max(e.max()/e.min() if e.min() > 0 else np.inf for e in evs)
    k_lv = kap(make_pipeline(LDA()), f, y)
    k_ts = kap(make_pipeline(TangentSpace(metric='riemann'), LDA()), covs, y)
    rows.append((dd, k_lv, k_ts, sep_of(covs, y), spread, cond))
    print(f"{dd:>6}d {k_lv:>9.3f} {k_ts:>7.3f} {sep_of(covs,y):>8.3f} {spread:>9.2e} "
          f"{cond:>9.1e} {time.time()-t0:>5.0f}", flush=True)

import pandas as pd
pd.DataFrame(rows, columns=["delta_deg","kappa_logvar","kappa_tangent","sep_ratio",
                            "feat_spread","cond_max"]).to_csv("results/step3_calib.csv", index=False)
best = min(rows, key=lambda r: abs(r[1] - 0.8))
print(f"\nclosest to kappa_logvar = 0.8 -> delta = {best[0]} deg (k_lv = {best[1]:.3f})")
print("wrote results/step3_calib.csv")
