"""Regenerate all figures with every annotation OUT of the data area."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
SURF, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]
RAMP = ["#86b6ef","#5598e7","#2a78d6","#256abf","#184f95","#0d366b"]
OUT = "../figures"

def style(ax, grid="y"):
    ax.set_facecolor(SURF); ax.grid(axis=grid, color="#ebeae6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    for s in ("left","bottom"): ax.spines[s].set_color("#d8d7d2")
    ax.tick_params(colors=INK2, labelsize=8.8, length=3)

def panel_title(ax, title, sub):
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="bold", pad=22)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")

# ---------------- 1. noise axis ----------------
d = pd.read_csv("results/step2d_corrected.csv"); v = d[d.valid]
ref = v[(v.frac==0.0) & (np.isinf(v.snr))].iloc[0]
norm = {"sep":ref.sep_ratio,"lv":ref.kappa_logvar,"ts":ref.kappa_tangent}
col = {"sep":"sep_ratio","lv":"kappa_logvar","ts":"kappa_tangent"}
LBL = {"sep":"separability index","lv":"decoder (geometry-independent)","ts":"decoder (Riemannian)"}
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURF)
fig.subplots_adjust(wspace=0.26, top=0.74, bottom=0.21, left=0.075, right=0.975)
ax = axes[0]
a = v[v.frac==0.0].copy(); a["x"] = np.where(np.isinf(a.snr), 16.0, a.snr); a = a.sort_values("x")
for k in ["sep","lv","ts"]:
    ax.plot(a.x, a[col[k]]/norm[k], "-o", lw=2, ms=5.5, color=CAT[["sep","lv","ts"].index(k)],
            mec=SURF, mew=1.5, label=LBL[k], clip_on=False, zorder=3)
ax.set_xscale("log", base=2); ax.set_xticks([0.25,0.5,1,2,4,8,16])
ax.set_xticklabels(["0.25","0.5","1","2","4","8","clean"]); ax.invert_xaxis()
ax.set_xlabel("observation SNR  (noisier $\\rightarrow$)", color=INK2, fontsize=9.5)
ax.set_ylabel("value relative to clean recording", color=INK2, fontsize=9.5)
panel_title(ax, "A.  index and both decoders fall together",
            "intact network; all three degrade in step as noise rises")
ax.axhline(1.0, color="#d8d7d2", lw=1, zorder=1); ax.set_ylim(-0.25, 1.12)
ax = axes[1]
for k, mk in [("lv","o"), ("ts","s")]:
    ax.scatter(v.sep_ratio, v[col[k]], s=34, c=CAT[["sep","lv","ts"].index(k)],
               edgecolors=SURF, linewidths=1.2, marker=mk, label=LBL[k], zorder=3)
ax.set_xscale("log")
ax.set_xlabel("separability index  (log scale)", color=INK2, fontsize=9.5)
ax.set_ylabel("decoder $\\kappa$", color=INK2, fontsize=9.5)
rr, _ = stats.spearmanr(v.sep_ratio, v.kappa_logvar)
panel_title(ax, "B.  across all 72 valid conditions",
            f"geometry-independent decoder: Spearman $\\rho$ = {rr:+.2f}")
ax.set_ylim(-0.25, 1.12)
for a_ in axes: style(a_)
axes[0].legend(frameon=False, fontsize=8.6, loc="upper left", bbox_to_anchor=(0, -0.20),
               ncol=3, labelcolor=INK2, handlelength=1.6, columnspacing=1.4)
fig.suptitle("The index tracks decodability under sensor noise", x=0.075, y=0.93,
             ha="left", color=INK, fontsize=12.5, fontweight="bold")
fig.savefig(f"{OUT}/01_noise_axis.png", dpi=200, facecolor=SURF)
plt.close(fig); print("1 ok")

# ---------------- 2. damage axis ----------------
d = pd.read_csv("results/step4_fine.csv"); v = d[d.valid]
fr = sorted(v.frac.unique())
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURF)
fig.subplots_adjust(wspace=0.26, top=0.74, bottom=0.21, left=0.075, right=0.975)
ax = axes[0]
for c, f in zip(RAMP, fr):
    s = v[v.frac==f]
    ax.scatter(s.sep_ratio, s.kappa_logvar, s=46, c=c, edgecolors=SURF, linewidths=1.3,
               label=f"{int(f*100)}%", zorder=3)
r, _ = stats.pearsonr(v.sep_ratio, v.kappa_logvar)
ax.set_xlabel("separability index", color=INK2, fontsize=9.5)
ax.set_ylabel("$\\kappa$, geometry-independent decoder", color=INK2, fontsize=9.5)
panel_title(ax, "A.  damage lowers both, together",
            f"all 48 conditions, r = {r:+.3f}")
leg = ax.legend(frameon=False, fontsize=8.0, title="pathways cut", loc="upper left",
                bbox_to_anchor=(0, -0.20), ncol=6, labelcolor=INK2,
                handletextpad=0.25, columnspacing=0.8)
leg.get_title().set_color(INK2); leg.get_title().set_fontsize(8.0)
ax = axes[1]
rs = []
for f in fr:
    s = v[v.frac==f]; rr, pp = stats.pearsonr(s.sep_ratio, s.kappa_logvar); rs.append(rr)
xs = np.arange(len(fr))
ax.vlines(xs, 0, rs, color="#d8d7d2", lw=1.5, zorder=1)
ax.scatter(xs, rs, s=84, c=RAMP, edgecolors=SURF, linewidths=1.5, zorder=3)
ax.set_xticks(xs); ax.set_xticklabels([f"{int(f*100)}%" for f in fr])
ax.set_xlabel("lesion dose held constant", color=INK2, fontsize=9.5)
ax.set_ylabel("r (index, $\\kappa$) within that dose", color=INK2, fontsize=9.5)
panel_title(ax, "B.  the relation survives with dose removed",
            "every dose significant at p < .05")
ax.set_ylim(0, 1.15); ax.axhline(0, color="#d8d7d2", lw=1)
for a_ in axes: style(a_)
fig.suptitle("Circuit damage: the index tracks what the damage cost", x=0.075, y=0.93,
             ha="left", color=INK, fontsize=12.5, fontweight="bold")
fig.savefig(f"{OUT}/02_damage_axis.png", dpi=200, facecolor=SURF)
plt.close(fig); print("2 ok")

# ---------------- 3. ensemble ----------------
d = pd.read_csv("results/step5_all_models.csv"); v = d[d.valid]
models = sorted(v.model.unique())
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), facecolor=SURF)
fig.subplots_adjust(wspace=0.26, top=0.74, bottom=0.21, left=0.075, right=0.975)
ax = axes[0]
for c, m in zip(CAT, models):
    g = v[v.model==m]
    ax.scatter(g.sep_ratio, g.kappa_logvar, s=34, c=c, edgecolors=SURF, linewidths=1.1,
               label=f"model {m.split('/')[-1]}", zorder=3)
r, _ = stats.pearsonr(v.sep_ratio, v.kappa_logvar)
ax.set_xlabel("separability index", color=INK2, fontsize=9.5)
ax.set_ylabel("$\\kappa$, geometry-independent decoder", color=INK2, fontsize=9.5)
panel_title(ax, "A.  three independently trained networks",
            f"pooled n = {len(v)}, r = {r:+.3f}")
ax.legend(frameon=False, fontsize=8.4, loc="upper left", bbox_to_anchor=(0, -0.20),
          ncol=3, labelcolor=INK2, handletextpad=0.3)
ax = axes[1]
w = 0.26
for i, (c, m) in enumerate(zip(CAT, models)):
    g = v[v.model==m]; xs = []; rr_ = []
    for j, f in enumerate(sorted(v.frac.unique())):
        s = g[g.frac==f]
        if len(s) > 3 and s.kappa_logvar.nunique() > 2:
            q, _ = stats.pearsonr(s.sep_ratio, s.kappa_logvar); xs.append(j+(i-1)*w); rr_.append(q)
    ax.vlines(xs, 0, rr_, color="#e4e3df", lw=1.2, zorder=1)
    ax.scatter(xs, rr_, s=62, c=c, edgecolors=SURF, linewidths=1.3, zorder=3)
ax.set_xticks(range(len(sorted(v.frac.unique()))))
ax.set_xticklabels([f"{int(f*100)}%" for f in sorted(v.frac.unique())])
ax.set_xlabel("lesion dose held constant", color=INK2, fontsize=9.5)
ax.set_ylabel("r (index, $\\kappa$) within that dose", color=INK2, fontsize=9.5)
panel_title(ax, "B.  within-dose tests", "15 of 15 significant at p < .05")
ax.set_ylim(0, 1.15); ax.axhline(0, color="#d8d7d2", lw=1)
for a_ in axes: style(a_)
fig.suptitle("The relationship is not one model's quirk", x=0.075, y=0.93,
             ha="left", color=INK, fontsize=12.5, fontweight="bold")
fig.savefig(f"{OUT}/03_ensemble_replication.png", dpi=200, facecolor=SURF)
plt.close(fig); print("3 ok")

# ---------------- 4. specificity ----------------
SHORT = {"T4_canonical_input":"T4 canonical input","onto_T5":"onto T5","onto_T4":"onto T4",
         "from_photoreceptor":"from photoreceptor","onto_T4_and_T5":"onto T4 + T5"}
d = pd.read_csv("results/step6_both_models.csv")
g = d.groupby(["model","condition","kind","n_cut"]).agg(sep=("sep_ratio","mean"),
                                                        k=("kappa_logvar","mean")).reset_index()
rows = []
for m in sorted(g.model.unique()):
    gm = g[g.model==m]
    for _, r_ in gm[gm.kind=="targeted"].iterrows():
        rnd = gm[gm.condition.str.startswith(f"random_match_{r_.condition}_s")]
        rows.append(dict(model=m, cond=r_.condition, n=int(r_.n_cut), sep=r_.sep, k=r_.k,
                         rk=rnd.k.mean(), rs=rnd.sep.mean(),
                         rk_lo=rnd.k.min(), rk_hi=rnd.k.max()))
R = pd.DataFrame(rows); R["dk"] = R.k - R.rk; R["ds"] = R.sep - R.rs
fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), facecolor=SURF)
fig.subplots_adjust(wspace=0.09, top=0.73, bottom=0.20, left=0.072, right=0.775)
ax = axes[0]
ax.axhspan(-1.3, 0, xmin=0, xmax=0.52, color="#eef4fc", zorder=0)
ax.axhspan(0, 1.3, xmin=0.52, xmax=1, color="#eef4fc", zorder=0)
ax.axhline(0, color="#d8d7d2", lw=1, zorder=1); ax.axvline(0, color="#d8d7d2", lw=1, zorder=1)
for c, m in zip(CAT, sorted(R.model.unique())):
    gm = R[R.model==m]
    ax.scatter(gm.ds, gm.dk, s=74, c=c, edgecolors=SURF, linewidths=1.4, zorder=3,
               label=f"model {m.split('/')[-1]}")
ax.set_xlabel("index:  targeted minus matched random", color=INK2, fontsize=9.5)
ax.set_ylabel("decoder $\\kappa$:  targeted minus matched random", color=INK2, fontsize=9.5)
panel_title(ax, "A.  10 of 10 agreement",
            "lower-left and upper-right shading: index and decoder concur")
ax.set_ylim(-1.3, 1.3)
ax.legend(frameon=False, fontsize=8.4, loc="upper left", bbox_to_anchor=(0, -0.19),
          ncol=2, labelcolor=INK2, handletextpad=0.3)
ax = axes[1]
R2 = R.sort_values(["model","n"]).reset_index(drop=True)
yp = np.arange(len(R2))[::-1]
for yy, (_, r_) in zip(yp, R2.iterrows()):
    ax.plot([r_.rk_lo, r_.rk_hi], [yy, yy], color="#c9c8c3", lw=5, solid_capstyle="round", zorder=2)
    ax.scatter([r_.rk], [yy], s=24, c=INK3, zorder=3)
    ax.scatter([r_.k], [yy], s=82, c=CAT[sorted(R.model.unique()).index(r_.model)],
               edgecolors=SURF, linewidths=1.4, zorder=4, marker="D")
ax.set_yticks(yp)
ax.set_yticklabels([f"{SHORT[r_.cond]}   ({r_.n})   model {r_.model.split('/')[-1]}"
                    for _, r_ in R2.iterrows()], fontsize=8.0, color=INK2)
ax.yaxis.tick_right()
ax.set_xlabel("decoder $\\kappa$", color=INK2, fontsize=9.5)
panel_title(ax, "B.  targeted lesion vs matched random range",
            "diamond: targeted    bar: range of 3 size-matched random lesions")
ax.set_xlim(-0.08, 1.08); ax.set_ylim(-0.7, len(R2)-0.3)
for i, a_ in enumerate(axes):
    style(a_, grid="x" if i == 1 else "y")
    if i == 1:
        a_.spines["left"].set_visible(False); a_.spines["right"].set_color("#d8d7d2")
        a_.tick_params(axis="y", length=0, pad=6)
fig.suptitle("Specificity: it flags damage that costs decodability, and only that",
             x=0.072, y=0.93, ha="left", color=INK, fontsize=12.5, fontweight="bold")
fig.savefig(f"{OUT}/04_specificity.png", dpi=200, facecolor=SURF)
plt.close(fig); print("4 ok")
