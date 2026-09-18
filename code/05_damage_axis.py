"""
step4_fine.py -- the lesion axis at a NON-SATURATED operating point, resolved
where the action actually is.

step2 could not answer "does the index track decodability as damage accumulates"
because the task was at ceiling (kappa = 1.000 everywhere). step3 showed the
usable knob is not class similarity -- at 2 deg separation kappa was still 1.000,
because a deterministic network has almost no within-class variability -- but
observation SNR. At SNR = 1.0 the intact network sits at kappa_logvar = 0.770.

step3 also showed everything floors beyond 20-30% lesion, so the coarse grid's
correlation was partly a floor effect. This run resolves 0-25% with 3 lesion
seeds per level and 3 noise seeds per condition.
"""
import os, sys, time
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

CL = [1, 2, 3, 4]
N_PER, N_FRAMES, DT = 15, 300, 1/100.
SPAT_FREQ, TEMP_FREQ, NOISE, TOP_K, BATCH = 0.18, 6.0, 0.05, 12, 5
ANGLES = [0.0, np.pi/4, np.pi/2, 3*np.pi/4]
SNR = 1.0
FRACS = [0.05, 0.10, 0.15, 0.20, 0.25]
LESION_SEEDS = [0, 1, 2]
NOISE_SEEDS = [0, 1, 2]
OUT = "results/step4_fine.csv"
y = np.repeat(CL, N_PER)

nv = NetworkView(str(flyvis.results_dir / "flow/0000/000"))
net = nv.init_network(); net.eval()
orig = net.edges_syn_strength.data.clone(); n_conn = len(orig)
df = net.connectome.nodes.to_df()
r1 = df[(df.role == 'input') & (df.type == 'R1')].sort_values('index')
uv = r1[['u', 'v']].to_numpy(); u, v = uv[:, 0].astype(float), uv[:, 1].astype(float)
XC, YC = u + v/2.0, v*np.sqrt(3)/2.0
central = np.array(net.connectome.central_cells_index)

def lv(X): return np.log(X.var(2) + 1e-30)
def kap(p, X, yy, s=0):
    return cohen_kappa_score(yy, cross_val_predict(p, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=s)))
def sep(c, yy):
    cm = {k: mean_riemann(c[yy == k]) for k in CL}
    wi = np.mean([np.mean([distance_riemann(C, cm[k]) for C in c[yy == k]]) for k in CL])
    be = np.mean([distance_riemann(cm[i], cm[j]) for i in CL for j in CL if i < j])
    return be / wi

rng0 = np.random.default_rng(0)
t = np.arange(N_FRAMES) * DT
movies = []
for ang in ANGLES:
    proj = XC * np.cos(ang) + YC * np.sin(ang)
    for _ in range(N_PER):
        ph = 2*np.pi*(SPAT_FREQ*proj[None, :] - TEMP_FREQ*t[:, None]) + rng0.uniform(0, 2*np.pi)
        m = 0.5 + 0.5*np.sin(ph) + rng0.normal(0, NOISE, (N_FRAMES, len(proj)))
        movies.append(np.clip(m, 0, 1).astype(np.float32))
movies = np.asarray(movies)

def simulate():
    o = []
    for i in range(0, len(movies), BATCH):
        with torch.no_grad():
            a = net.simulate(torch.from_numpy(movies[i:i+BATCH])[:, :, None, :], dt=DT)
        o.append(a[:, :, central].numpy().transpose(0, 2, 1)); del a
    return np.concatenate(o).astype(np.float64)

os.makedirs("results/act4", exist_ok=True)
# Re-derive the readout exactly as step2b did: simulate the INTACT network at all
# 65 central cells, keep the top-12 by variance. The saved step2b activity is
# already reduced, so it cannot say WHICH 12 -- and scoring lesions on 65 channels
# against a 12-channel intact reference would compare different quantities.
A0_full = simulate()
keep = np.argsort(-A0_full.var(2).mean(0))[:TOP_K]
A0 = A0_full[:, keep, :]
print(f"[setup] SNR={SNR}  readout re-derived: {TOP_K} of {A0_full.shape[1]} central cells", flush=True)
_chk = np.load("results/act/act_f0.0_s-1.npy").astype(np.float64)
print(f"[setup] reproduces step2b intact activity: {np.allclose(A0, _chk, rtol=1e-4, atol=1e-5)}", flush=True)

# guard
gr = np.random.default_rng(0)
kn = [kap(make_pipeline(LDA()), lv(gr.normal(0, 1, A0.shape)), y, s) for s in range(5)]
print(f"[guard] structureless-input kappa mean {np.mean(kn):+.3f} {np.round(kn,3).tolist()}", flush=True)
if abs(np.mean(kn)) > 0.15:
    sys.exit("!! aborting: decoder scores off-chance on structureless input")

rows = []
def score(A, frac, lseed):
    r = A.std()
    for ns in NOISE_SEEDS:
        X = A + np.random.default_rng(100 + ns).normal(0, r/SNR, A.shape)
        ev = np.linalg.eigvalsh(np.array([np.cov(x) for x in X]))
        cond = max(e.max()/e.min() if e.min() > 0 else np.inf for e in ev)
        rk = min(np.linalg.matrix_rank(np.cov(x)) for x in X)
        covs = Covariances(estimator='oas').transform(X)
        rows.append(dict(frac=frac, lesion_seed=lseed, noise_seed=ns,
                         sep_ratio=sep(covs, y),
                         kappa_logvar=kap(make_pipeline(LDA()), lv(X), y, ns),
                         kappa_tangent=kap(make_pipeline(TangentSpace(metric='riemann'), LDA()), covs, y, ns),
                         cond_max=float(cond), valid=bool(rk == TOP_K and cond < 1e8)))

score(A0, 0.0, -1)
for frac in FRACS:
    for ls in LESION_SEEDS:
        t0 = time.time()
        cut = np.random.default_rng(1000 + ls).choice(n_conn, size=int(frac*n_conn), replace=False)
        net.edges_syn_strength.data.copy_(orig); net.edges_syn_strength.data[cut] = 0.0
        A = simulate()
        net.edges_syn_strength.data.copy_(orig)
        np.save(f"results/act4/act_f{frac:.2f}_s{ls}.npy", A[:, keep, :].astype(np.float32))
        score(A[:, keep, :], frac, ls)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        sub = [r for r in rows if r['frac'] == frac and r['lesion_seed'] == ls]
        print(f"  frac={frac:.2f} ls={ls}  sep={np.mean([s['sep_ratio'] for s in sub]):6.3f}  "
              f"k_lv={np.mean([s['kappa_logvar'] for s in sub]):+.3f}  "
              f"k_tan={np.mean([s['kappa_tangent'] for s in sub]):+.3f}  [{time.time()-t0:.0f}s]", flush=True)
print("\nwrote", OUT, flush=True)
