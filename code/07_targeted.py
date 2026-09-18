#!/usr/bin/env python3
"""
step6_targeted.py -- targeted vs random lesions at matched dose.

THE QUESTION
------------
Steps 4-5 cut random pathways and found the index tracks decodability
(r = +0.93 pooled, 15/15 within-dose). But "which pathways you cut matters more
than how many" was the largest effect in that data. So: if you cut the pathways
that actually carry the task -- the motion pathway -- does the index register
that as worse than cutting the SAME NUMBER of random pathways?

If it does, the index is not merely counting damage; it is reading what the
damage cost. That is the claim an attribution argument actually needs.

PATHWAY INDEXING -- verified, not assumed
-----------------------------------------
`edges_syn_strength` has one entry per (source_type, target_type) pair. flyvis
builds it with `edges.groupby([...], as_index=False, sort=False)`
(initialization.py:496-497), so the order is FIRST APPEARANCE in the edge table,
NOT sorted order. Reproduced here with the same call, then verified STRUCTURALLY:
`edges_sign` is grouped identically and its per-pathway sign is a fixed property
of the connectome, so the sign vector derived from our ordering must equal
flyvis's own `edges_sign` element for element. A shuffled ordering fails this
test, which is what gives it power.

An earlier version verified this behaviourally -- zero (R1 -> L1) and check L1
moves most. That passed on model 000 and FALSELY FAILED on model 001, because L1
is driven by R1-R6 and removing one sixth of its drive is not always the largest
downstream change. A check that fires when nothing is wrong is as bad as one that
never fires; the structural test has neither failure mode and needs no simulation.

RESUMABLE
---------
Every condition is written to the CSV as soon as it finishes. On restart, any
condition already complete in the CSV is skipped. Close the laptop whenever;
re-run the same command to continue.
"""
import os, sys, time, json
import numpy as np, pandas as pd, torch
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

MODEL      = os.environ.get("FLYVIS_MODEL", "flow/0000/000")
TAG        = MODEL.replace("/", "_")
CL         = [1, 2, 3, 4]
N_PER      = 15
N_FRAMES   = 300
DT         = 1/100.
SPAT_FREQ, TEMP_FREQ, NOISE = 0.18, 6.0, 0.05
ANGLES     = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
TOP_K      = 12
BATCH      = 5
SNR        = 1.0
NOISE_SEEDS = [0, 1, 2]
RANDOM_SEEDS = [0, 1, 2]
OUTDIR     = "results"
OUT        = f"{OUTDIR}/step6_targeted_{TAG}.csv"
ACT        = f"{OUTDIR}/act6_{TAG}"
y = np.repeat(CL, N_PER)


# ---------------------------------------------------------------- helpers
def lv(X):
    return np.log(X.var(2) + 1e-30)

def kap(pipe, X, yy, seed=0):
    return cohen_kappa_score(yy, cross_val_predict(
        pipe, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=seed)))

def sep(covs, yy):
    cm = {k: mean_riemann(covs[yy == k]) for k in CL}
    wi = np.mean([np.mean([distance_riemann(C, cm[k]) for C in covs[yy == k]]) for k in CL])
    be = np.mean([distance_riemann(cm[i], cm[j]) for i in CL for j in CL if i < j])
    return be / wi


def main():
    os.makedirs(OUTDIR, exist_ok=True); os.makedirs(ACT, exist_ok=True)

    # ---- device: try MPS, fall back to CPU if flyvis can't run on it --------
    dev = "cpu"
    if torch.backends.mps.is_available():
        try:
            _ = (torch.zeros(8, device="mps") + 1).sum().item()
            dev = "mps"
        except Exception as ex:
            print(f"[device] MPS present but unusable ({ex}); using CPU", flush=True)
    print(f"[device] {dev} | torch {torch.__version__} | threads {torch.get_num_threads()}", flush=True)
    print(f"[model] {MODEL}  -> {OUT}", flush=True)

    nv = NetworkView(str(flyvis.results_dir / MODEL))
    net = nv.init_network(); net.eval()
    if dev == "mps":
        try:
            net = net.to("mps")
            _t = torch.rand(1, 20, 1, 721, device="mps")
            with torch.no_grad():
                net.simulate(_t, dt=DT)
            print("[device] MPS forward pass OK", flush=True)
        except Exception as ex:
            print(f"[device] MPS forward failed ({type(ex).__name__}); falling back to CPU", flush=True)
            net = net.to("cpu"); dev = "cpu"

    orig = net.edges_syn_strength.data.clone()
    edges = net.connectome.edges.to_df()
    nodes = net.connectome.nodes.to_df()

    # ---- pathway index map, reproduced exactly then VERIFIED ---------------
    grouped = edges.groupby(['source_type', 'target_type'], as_index=False, sort=False)
    pairs = grouped.size()[['source_type', 'target_type']].reset_index(drop=True)
    assert len(pairs) == len(orig), "pair count != syn_strength length"

    r1 = nodes[(nodes.role == 'input') & (nodes.type == 'R1')].sort_values('index')
    uv = r1[['u', 'v']].to_numpy()
    u_, v_ = uv[:, 0].astype(float), uv[:, 1].astype(float)
    XC, YC = u_ + v_/2.0, v_*np.sqrt(3)/2.0
    central = np.array(net.connectome.central_cells_index)
    ctypes = nodes.set_index('index').loc[central, 'type'].to_numpy()

    def verify_mapping():
        """Exact, model-independent: our group ordering must reproduce edges_sign."""
        mysign = grouped['sign'].first()['sign'].to_numpy()
        libsign = net.edges_sign.data.cpu().numpy()
        ok = bool(np.array_equal(mysign, libsign))
        perm = np.random.default_rng(0).permutation(len(mysign))
        powered = not bool(np.array_equal(mysign[perm], libsign))
        print(f"[verify] sign vector matches flyvis edges_sign: {ok} "
              f"({int((mysign != libsign).sum())} mismatches of {len(mysign)}); "
              f"shuffled-order control fails as required: {powered}", flush=True)
        return ok and powered

    if not verify_mapping():
        sys.exit("!! pathway index mapping failed verification -- refusing to run. "
                 "Lesioning the wrong pathways would produce clean-looking nonsense.")

    # ---- condition definitions --------------------------------------------
    st, tt = pairs.source_type.values, pairs.target_type.values
    is_T4 = np.array([str(x).startswith('T4') for x in tt])
    is_T5 = np.array([str(x).startswith('T5') for x in tt])
    from_R = np.array([bool(str(x) and str(x)[0] == 'R' and str(x)[1:].isdigit()) for x in st])
    T4_CANON = {'Mi1', 'Tm3', 'Mi4', 'Mi9'}
    canon = np.array([(str(x) in T4_CANON) for x in st]) & is_T4

    targeted = {
        "onto_T4":            np.where(is_T4)[0],
        "onto_T5":            np.where(is_T5)[0],
        "onto_T4_and_T5":     np.where(is_T4 | is_T5)[0],
        "from_photoreceptor": np.where(from_R)[0],
        "T4_canonical_input": np.where(canon)[0],
    }
    conditions = [("intact", np.array([], dtype=int))]
    for name, idx in targeted.items():
        conditions.append((name, idx))
    for name, idx in targeted.items():
        for s in RANDOM_SEEDS:
            cut = np.random.default_rng(5000 + s + 17*len(idx)).choice(
                len(orig), size=len(idx), replace=False)
            conditions.append((f"random_match_{name}_s{s}", cut))
    print("[plan] " + ", ".join(f"{n}({len(i)})" for n, i in conditions[:6]), flush=True)
    print(f"[plan] {len(conditions)} conditions total", flush=True)

    # ---- resume ------------------------------------------------------------
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        for c, g in prev.groupby("condition"):
            if len(g) == len(NOISE_SEEDS):
                done.add(c)
        print(f"[resume] {len(done)} conditions already complete, skipping them", flush=True)
    rows = pd.read_csv(OUT).to_dict("records") if os.path.exists(OUT) else []

    # ---- stimuli (identical for every condition) ---------------------------
    rng0 = np.random.default_rng(0)
    tarr = np.arange(N_FRAMES) * DT
    movies = []
    for ang in ANGLES:
        proj = XC*np.cos(ang) + YC*np.sin(ang)
        for _ in range(N_PER):
            ph = 2*np.pi*(SPAT_FREQ*proj[None, :] - TEMP_FREQ*tarr[:, None]) + rng0.uniform(0, 2*np.pi)
            m = 0.5 + 0.5*np.sin(ph) + rng0.normal(0, NOISE, (N_FRAMES, len(proj)))
            movies.append(np.clip(m, 0, 1).astype(np.float32))
    movies = np.asarray(movies)

    def simulate():
        o = []
        for i in range(0, len(movies), BATCH):
            chunk = torch.from_numpy(movies[i:i+BATCH])[:, :, None, :].to(dev)
            with torch.no_grad():
                a = net.simulate(chunk, dt=DT)
            o.append(a[:, :, central].cpu().numpy().transpose(0, 2, 1)); del a
        return np.concatenate(o).astype(np.float64)

    # readout fixed from intact
    kp_file = f"{OUTDIR}/step6_keep_{TAG}.json"
    if os.path.exists(kp_file):
        keep = np.array(json.load(open(kp_file)))
        print(f"[readout] loaded fixed readout from {kp_file}", flush=True)
        A0 = None
    else:
        A0_full = simulate()
        keep = np.argsort(-A0_full.var(2).mean(0))[:TOP_K]
        json.dump([int(k) for k in keep], open(kp_file, "w"))
        A0 = A0_full[:, keep, :]
        print(f"[readout] {list(ctypes[keep])}", flush=True)

    # guard that can actually fail
    probe = A0 if A0 is not None else np.load(f"{ACT}/intact.npy").astype(np.float64)
    gr = np.random.default_rng(0)
    kn = [kap(make_pipeline(LDA()), lv(gr.normal(0, 1, probe.shape)), y, s) for s in range(5)]
    print(f"[guard] structureless-input kappa mean {np.mean(kn):+.3f}", flush=True)
    if abs(np.mean(kn)) > 0.15:
        sys.exit("!! decoder scores off-chance on structureless input -- aborting")

    # ---- run ---------------------------------------------------------------
    for name, cut in conditions:
        if name in done:
            continue
        t0 = time.time()
        if name == "intact" and A0 is not None:
            A = A0
        else:
            net.edges_syn_strength.data.copy_(orig)
            if len(cut):
                net.edges_syn_strength.data[torch.as_tensor(cut, device=net.edges_syn_strength.device)] = 0.0
            A = simulate()[:, keep, :]
            net.edges_syn_strength.data.copy_(orig)
        np.save(f"{ACT}/{name}.npy", A.astype(np.float32))
        scale = A.std()
        for ns in NOISE_SEEDS:
            X = A + np.random.default_rng(100 + ns).normal(0, scale/SNR, A.shape)
            ev = np.linalg.eigvalsh(np.array([np.cov(x) for x in X]))
            cond = max(e.max()/e.min() if e.min() > 0 else np.inf for e in ev)
            rk = min(np.linalg.matrix_rank(np.cov(x)) for x in X)
            covs = Covariances(estimator='oas').transform(X)
            rows.append(dict(condition=name, kind=("intact" if name == "intact"
                                                   else "random" if name.startswith("random") else "targeted"),
                             n_cut=int(len(cut)), noise_seed=ns,
                             sep_ratio=sep(covs, y),
                             kappa_logvar=kap(make_pipeline(LDA()), lv(X), y, ns),
                             kappa_tangent=kap(make_pipeline(TangentSpace(metric='riemann'), LDA()), covs, y, ns),
                             cond_max=float(cond), valid=bool(rk == TOP_K and cond < 1e8)))
        pd.DataFrame(rows).to_csv(OUT, index=False)
        sub = [r for r in rows if r["condition"] == name]
        print(f"  {name:34s} n_cut={len(cut):3d}  sep={np.mean([s['sep_ratio'] for s in sub]):6.3f}  "
              f"k_lv={np.mean([s['kappa_logvar'] for s in sub]):+.3f}  [{time.time()-t0:.0f}s]", flush=True)

    print(f"\nwrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
