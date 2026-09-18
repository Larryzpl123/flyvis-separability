"""
probe_flyvis_separability.py -- FEASIBILITY ONLY. Not a scientific result.

THE ONE QUESTION
----------------
Can a connectome-constrained flyvis network produce activity that a
Riemannian separability index can eat, with a covariance that is
genuinely SPD rather than rescued by shrinkage?

WHY THE FIRST VERSION OF THIS FILE WAS WRONG -- READ THIS
---------------------------------------------------------
v1 used an UNTRAINED network and checked SPD with `min_eigenvalue > 0`.
It printed PASS. That was a false pass:
    min_eig was +1e-17 (numerically zero) and the raw covariance rank was
    27-29 of 65, because 33 of the 65 central cells had EXACTLY zero variance
    in an untrained net (relu + random init -> dead units).
A finite number out of OAS shrinkage is not the same as an invariant one --
the same point that applies to any shrinkage-rescued covariance.
So this version (a) uses the PRETRAINED ensemble and (b) judges SPD by
CONDITION NUMBER, not by sign of the smallest eigenvalue.

separability_ratio() is held byte-for-byte identical across every script in
this repository. Do not tidy it.
"""
import sys, time
import numpy as np
import torch

from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

import flyvis
from flyvis import NetworkView

CLASSES   = [1, 2, 3, 4]
ANGLES    = {1: 0.0, 2: np.pi/4, 3: np.pi/2, 4: 3*np.pi/4}
N_PER_CLS = 12
N_FRAMES  = 300
DT        = 1/100.
SPAT_FREQ = 0.18
TEMP_FREQ = 6.0
NOISE     = 0.05
SEED      = 0
MODEL     = "flow/0000/000"   # pretrained; flyvis download-pretrained
ZSCORE    = True                # channel variances span ~7 orders of magnitude
COND_MAX  = 1e8                 # above this the covariance is singular in practice
TOP_K     = 12                  # readout units; >16 and the covariance goes singular


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


def hexal_xy(uv):
    u, v = uv[:, 0].astype(float), uv[:, 1].astype(float)
    return u + v / 2.0, v * np.sqrt(3) / 2.0


def make_movie(xy, angle, n_frames, rng):
    x, y = xy
    proj = x * np.cos(angle) + y * np.sin(angle)
    phase0 = rng.uniform(0, 2 * np.pi)
    t = np.arange(n_frames) * DT
    ph = 2*np.pi*(SPAT_FREQ*proj[None, :] - TEMP_FREQ*t[:, None]) + phase0
    mov = 0.5 + 0.5*np.sin(ph) + rng.normal(0, NOISE, (n_frames, len(proj)))
    return np.clip(mov, 0, 1).astype(np.float32)


def cond_of(C):
    ev = np.linalg.eigvalsh(C)
    return (ev.max()/ev.min()) if ev.min() > 0 else np.inf, ev.min()


def main():
    rng = np.random.default_rng(SEED)
    nv = NetworkView(str(flyvis.results_dir / MODEL))
    net = nv.init_network(); net.eval()
    print(f"[net] pretrained {MODEL}", flush=True)

    df = net.connectome.nodes.to_df()
    r1 = df[(df.role == 'input') & (df.type == 'R1')].sort_values('index')
    xy = hexal_xy(r1[['u', 'v']].to_numpy())
    central = np.array(net.connectome.central_cells_index)
    n_units = len(central)
    if N_FRAMES <= n_units:
        sys.exit(f"!! N_FRAMES ({N_FRAMES}) must exceed n_units ({n_units}).")

    X, y = [], []
    t0 = time.time()
    for c in CLASSES:
        for _ in range(N_PER_CLS):
            inp = torch.from_numpy(make_movie(xy, ANGLES[c], N_FRAMES, rng))[None, :, None, :]
            with torch.no_grad():
                act = net.simulate(inp, dt=DT)
            X.append(act[0][:, central].numpy().T)
            y.append(c)
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
    print(f"[sim] {len(y)} trials in {time.time()-t0:.1f}s | X (trials,units,time) = {X.shape}")

    if ZSCORE:
        X = (X - X.mean(2, keepdims=True)) / (X.std(2, keepdims=True) + 1e-12)

    # 65 channels exceeds the intrinsic dimensionality of the response; keep the
    # TOP_K most variable units or the covariance is singular (see header).
    if TOP_K and TOP_K < n_units:
        keep = np.argsort(-np.asarray([np.cov(x).trace() for x in X]).mean()*0 -
                          X.var(2).mean(0))[:TOP_K]
        X = X[:, keep, :]
        n_units = TOP_K
        print(f"[readout] reduced to TOP_K={TOP_K} units")

    raw = np.array([np.cov(x) for x in X])
    conds = np.array([cond_of(c)[0] for c in raw])
    ranks = np.array([np.linalg.matrix_rank(c) for c in raw])
    mineig = min(cond_of(c)[1] for c in raw)
    covs = Covariances(estimator='oas').transform(X)

    print("\n=== SPD / conditioning (raw, no shrinkage) ===")
    print(f"  rank        : {ranks.min()}-{ranks.max()} of {n_units}")
    print(f"  min eigval  : {mineig:+.3e}")
    print(f"  cond number : median {np.median(conds):.2e}  worst {conds.max():.2e}")

    r_t, b_t, w_t = separability_ratio(covs, y)
    r_s, b_s, w_s = separability_ratio(covs, rng.permutation(y))
    gain = r_t/r_s if r_s else float('inf')
    print("\n=== separability index (phase2 function, unmodified) ===")
    print(f"  true      sep_ratio={r_t:.4f}  between={b_t:.4f}  within={w_t:.4f}")
    print(f"  shuffled  sep_ratio={r_s:.4f}  between={b_s:.4f}  within={w_s:.4f}")
    print(f"  true/shuffled = {gain:.2f}x")

    ok_rank = ranks.min() == n_units
    ok_cond = conds.max() < COND_MAX
    ok_move = gain > 1.5
    print("\n=== VERDICT ===")
    print(f"  [{'PASS' if ok_rank else 'FAIL'}] full rank on every trial")
    print(f"  [{'PASS' if ok_cond else 'FAIL'}] condition number < {COND_MAX:.0e} "
          f"(worst {conds.max():.1e})")
    print(f"  [{'PASS' if ok_move else 'FAIL'}] index distinguishes true from shuffled labels")
    print("\n  FEASIBILITY ONLY: hand-built gratings, one pretrained model, "
          f"n={N_PER_CLS}/class. No scientific claim.")
    return 0 if (ok_rank and ok_cond and ok_move) else 1


if __name__ == "__main__":
    sys.exit(main())
