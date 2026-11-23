"""Publication figures of the paper and of the results document, generated from the result files (no number typed by hand).
Four figures of 2 x 4 panels, written to report/ as vector PDF and 300-dpi PNG. Colour-blind-safe palette (Tol 'bright'),
one marker and one line style per method, shared method legend above each figure."""
import os, sys, glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "scripts"))
from analyze_grid import L, add
REP = os.path.join(HERE, "report"); RES = os.path.join(HERE, "results"); LLM = os.path.join(RES, "llm")
def LX(s): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "extra", f"{s}_c*.csv")))], ignore_index=True)

BLUE, CYAN, GREEN, YELLOW, RED, PURPLE, GREY, INK = "#4477AA", "#66CCEE", "#228833", "#CCBB44", "#EE6677", "#AA3377", "#9A9A9A", "#333333"
# internal name -> (display name, colour, marker, line style)
M = {"EmerKit": ("EmerKit", BLUE, "o", "-"), "Oracle": ("Oracle", GREY, "*", ":"), "RAG-MCP": ("RAG-MCP", RED, "^", "--"),
     "RAG-MCP+filter": ("RAG-MCP+F", PURPLE, "v", "-."), "Zeng": ("QoS-IP", YELLOW, "s", "--"), "Zeng+filter": ("QoS-IP+F", GREEN, "D", "-"),
     "EmerKit-no-interactions": ("EmerKit-NI", CYAN, "P", ":"), "EmerKit-declared": ("EmerKit-DA", CYAN, "X", ":"),
     "EmerKit-nosub": ("EmerKit-NS", CYAN, "P", ":"), "EmerKit-xi0.10": (r"EmerKit ($\xi$=0.10)", CYAN, "h", "-"),
     "Gen+RAG-MCP": ("RAG-MCP", RED, "^", "--"), "Gen+RAG-MCP+filter": ("RAG-MCP+F", PURPLE, "v", "-."), "Gen+Zeng": ("QoS-IP", YELLOW, "s", "--"),
     "Gen+Zeng+filter": ("QoS-IP+F", GREEN, "D", "-"), "Random": ("Random", GREY, "x", ":")}
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.4, "axes.labelsize": 7, "legend.fontsize": 6.2, "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "lines.linewidth": 1.25, "lines.markersize": 3.8, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "savefig.dpi": 300})

def new(rows=2):
    fig, ax = plt.subplots(rows, 4, figsize=(7.15, 1.78 * rows + 0.28)); return fig, ax
def ttl(a, letter, title): a.set_title(f"({letter}) {title}", loc="left", pad=3)
def ln(a, df, x, y, methods, pct=True):
    for m in methods:
        g = df[df.method == m].groupby(x)[y].mean(); n, c, mk, ls = M[m]
        a.plot(g.index, (100 if pct else 1) * g.values, color=c, marker=mk, ls=ls, label=n, markeredgecolor="white", markeredgewidth=0.4)
def toplegend(fig, methods, extra=()):
    h = [Line2D([0], [0], color=M[m][1], marker=M[m][2], ls=M[m][3], label=M[m][0], markeredgecolor="white", markeredgewidth=0.4) for m in methods] + list(extra)
    fig.legend(handles=h, loc="upper center", ncol=len(h), fontsize=6.4, columnspacing=1.3, handlelength=2.2, bbox_to_anchor=(0.5, 1.0))
def save(fig, name):
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=1.1, w_pad=0.9)
    fig.savefig(os.path.join(REP, name + ".pdf")); fig.savefig(os.path.join(REP, name + ".png"), dpi=300); plt.close(fig); print("wrote", name)
def val(a, x, y, s, **kw): a.text(x, y, s, ha="center", va="bottom", fontsize=5.6, **kw)
def heat(a, mat, rows, cols, fmt="{:.0f}", cmap="YlGnBu", vmin=None, vmax=None, cbar=None):
    im = a.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax); a.grid(False)
    a.set_xticks(range(len(cols))); a.set_xticklabels(cols); a.set_yticks(range(len(rows))); a.set_yticklabels(rows, fontsize=5.8)
    mid = (np.nanmax(mat) + np.nanmin(mat)) / 2 if vmin is None else (vmin + vmax) / 2
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            a.text(j, i, fmt.format(mat[i, j]), ha="center", va="center", fontsize=5.4, color="white" if mat[i, j] > mid else INK)
    for s_ in a.spines.values(): s_.set_visible(False)
    a.tick_params(length=0)

A1 = lambda: add(pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "a1_seed*.csv"))]))
SIX = ["EmerKit", "RAG-MCP", "RAG-MCP+filter", "Zeng", "Zeng+filter", "Oracle"]

# ============================================================== Figure 1: quality of toolkits (RQ1)
def fig_rq1():
    fig, ax = new()
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(LLM, "A5real_*.csv"))]).reset_index(drop=True)
    import run_probes_real as RP
    from emerkit.prep import Prepared
    T, _ = RP.design(); P = Prepared(T, RP.toucan_catalogue(0)[1], seed=0)
    err = d[(d.correct == 0) & d.chosen.notna()]
    clone = np.array([T[int(t)]["name"] == T[int(c)]["name"] and P.desc_sim(int(t), int(c)) >= 0.95 for t, c in zip(err.target, err.chosen)])
    d["functional"] = d.correct; d.loc[err.index[clone], "functional"] = 1
    models = [("claude-haiku-5-5", "Haiku"), ("claude-opus-5-5", "Opus")]
    a = ax[0, 0]; w = 0.26
    for j, (lab, col, hatch, f) in enumerate([("random", GREY, "", lambda g: g[g.confusable == 0].correct.mean()),
                                              ("confusable", RED, "//", lambda g: g[g.confusable == 1].correct.mean()),
                                              ("confusable,\nclones accepted", YELLOW, "..", lambda g: g[g.confusable == 1].functional.mean())]):
        v = [100 * f(d[d.model == m]) for m, _ in models]
        r = a.bar(np.arange(2) + (j - 1) * w, v, w, color=col, label=lab, hatch=hatch, edgecolor="white", linewidth=0.3)
        for x, y in zip(r, v): val(a, x.get_x() + w / 2, y, f"{y:.0f}")
    a.set_xticks(range(2)); a.set_xticklabels([n for _, n in models]); a.set_ylim(70, 128); a.set_yticks([70, 80, 90, 100]); a.set_ylabel("Correct choices (%)")
    a.legend(loc="upper center", fontsize=5.4, ncol=2, columnspacing=0.6, handlelength=1.1, title="Distractors", title_fontsize=5.6); ttl(a, "a", "Probes on real tools")
    a = ax[0, 1]; rows_, mat = [], []
    for m, n in models:
        for c, lab in [(0, "rand."), (1, "conf.")]:
            g = d[(d.model == m) & (d.confusable == c)].groupby("k").correct.mean(); rows_.append(f"{n}, {lab}"); mat.append(100 * g.values)
    heat(a, np.array(mat), rows_, [4, 8, 16, 32, 64], vmin=70, vmax=100); a.set_xlabel("Toolkit size $k$ (real tools)"); ttl(a, "b", "Correct choices (%) by size")
    a = ax[0, 2]; s6 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(LLM, "A6real_*.csv"))]); g = s6.groupby("method").agg(s=("success", "mean"), c=("clone_pairs", "mean"))
    ms = ["EmerKit", "RAG-MCP", "Random"]; b = a.bar(range(3), [100 * g.loc[m, "s"] for m in ms], color=[M[m][1] for m in ms], width=0.6)
    for i, m in enumerate(ms): a.text(i, 4, f"{100 * g.loc[m, 's']:.1f}", ha="center", color="white", fontsize=5.8, fontweight="bold")
    a2 = a.twinx(); a2.plot(range(3), [g.loc[m, "c"] for m in ms], color=INK, marker="D", ls="--", lw=0.9, ms=3.5); a2.set_ylim(0, 2.4); a2.set_ylabel("Clone pairs per toolkit", fontsize=6.2)
    a2.spines["right"].set_visible(True); a2.grid(False); a.set_xticks(range(3)); a.set_xticklabels([M[m][0] for m in ms], fontsize=5.8); a.set_ylim(0, 95); a.set_ylabel("Successful choices (%)")
    ttl(a, "c", "Agent on composed toolkits")
    a = ax[0, 3]; w = 0.26; r7 = {s: pd.read_csv(os.path.join(LLM, f"A7_{s}.csv")).groupby("method").recall.mean() for s in ["warm", "cold"]}
    for j, (m, col, hatch, lab) in enumerate([("EmerKit (no traces)", CYAN, "", "EmerKit, no traces"), ("EmerKit (traces)", BLUE, "//", "EmerKit, traces"), ("RAG-MCP", RED, "..", "RAG-MCP")]):
        v = [100 * r7[s][m] for s in ["warm", "cold"]]; r = a.bar(np.arange(2) + (j - 1) * w, v, w, color=col, label=lab, hatch=hatch, edgecolor="white", linewidth=0.3)
        for x, y in zip(r, v): val(a, x.get_x() + w / 2, y, f"{y:.1f}")
    a.set_xticks(range(2)); a.set_xticklabels(["history", "no history"]); a.set_ylabel("Recall of called tools (%)"); a.set_ylim(0, 31); a.set_yticks([0, 5, 10, 15, 20])
    a.legend(loc="upper right", fontsize=5.4, handlelength=1.1); ttl(a, "d", "Real trajectories")
    core = ["EmerKit", "EmerKit-no-interactions", "RAG-MCP", "Zeng"]
    a = ax[1, 0]; d2 = add(L("A2")); d2["cp_e"] = np.where(d2["eq"] == 1, (d2["cp"] > 0).astype(float), np.nan); ln(a, d2, "rho_conf", "cp_e", core)
    a.set_xticks([0, 0.1, 0.2, 0.3, 0.4]); a.set_xlabel("Capabilities with a near-duplicate"); a.set_ylabel("Toolkits with a\nnear-duplicate pair (%)"); ttl(a, "e", "Near-duplicates")
    a = ax[1, 1]; d3 = add(L("A3")); d3 = d3[d3.length_factor == 1]; ln(a, d3, "B", "over_e", core)
    a.set_xscale("log", base=2); a.set_xticks([500, 1000, 2000, 4000]); a.set_xticklabels(["0.5k", "1k", "2k", "4k"]); a.set_xlabel("Context budget $B$ (tokens)")
    a.set_ylabel(r"Toolkits above $\theta B$ (%)"); ttl(a, "f", "Context load")
    a1 = A1(); e = a1[a1["eq"] == 1]; order = ["EmerKit", "RAG-MCP", "RAG-MCP+filter", "Zeng", "Zeng+filter"]
    a = ax[1, 2]; bp = a.boxplot([100 * e[e.method == m].lb.values for m in order], patch_artist=True, widths=0.55, showfliers=False, medianprops=dict(color=INK, lw=0.9),
                                 whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.5))
    for p_, m in zip(bp["boxes"], order): p_.set_facecolor(M[m][1]); p_.set_alpha(0.85)
    a.set_xticks(range(1, 6)); a.set_xticklabels([M[m][0] for m in order], rotation=28, ha="right", fontsize=5.8); a.set_ylabel("Context used (% of $B$)"); ttl(a, "g", "Load distribution")
    a = ax[1, 3]; red = e.groupby("method").red.mean(); a.bar(range(5), [red[m] for m in order], color=[M[m][1] for m in order], width=0.6)
    for i, m in enumerate(order): val(a, i, red[m], f"{red[m]:.2f}")
    a.set_xticks(range(5)); a.set_xticklabels([M[m][0] for m in order], rotation=28, ha="right", fontsize=5.8); a.set_ylabel("Redundant providers per task"); a.set_ylim(0, 2.7); ttl(a, "h", "Redundancy")
    toplegend(fig, core); save(fig, "fig_rq1")

# ============================================================== Figure 2: norm compliance and guarantee (RQ2)
def fig_rq2():
    fig, ax = new(); a1 = A1(); g = a1.groupby("method").agg(eq=("eq", "mean"), dfr=("dfr_e", "mean")); a = ax[0, 0]
    for m in SIX + ["EmerKit-xi0.10"]:
        n, c, mk, _ = M[m]; a.scatter(100 * g.loc[m, "dfr"], 100 * g.loc[m, "eq"], s=34, color=c, marker=mk, zorder=3, edgecolor="white", linewidth=0.4)
    a.axvline(5, color=INK, ls=":", lw=0.7); a.text(6.5, 62, r"$\xi$ = 5%", fontsize=5.6); a.annotate(r"$\xi$=0.10", (100 * g.loc["EmerKit-xi0.10", "dfr"], 100 * g.loc["EmerKit-xi0.10", "eq"]), xytext=(4, -8), textcoords="offset points", fontsize=5.4)
    a.set_xlabel("Toolkits with a denied flow (%)"); a.set_ylabel("Tasks equipped (%)"); a.set_xlim(-4, 86); a.set_ylim(60, 104); ttl(a, "a", "Coverage against risk")
    tr = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "b1_trace_seed*.csv"))]).groupby("eps")[["test_eq", "test_dfr"]].mean(); a = ax[0, 1]
    a.plot(tr.index, 100 * tr.test_eq, color=BLUE, marker="o", ms=2.2, label="equipped"); a.plot(tr.index, 100 * tr.test_dfr, color=RED, marker="^", ms=2.2, ls="--", label="denied-flow rate")
    a.axhline(5, color=INK, ls=":", lw=0.7); a.set_xscale("symlog", linthresh=0.01); a.set_xticks([0, 0.01, 0.1, 0.5]); a.set_xticklabels(["0", "0.01", "0.1", "0.5"])
    a.set_xlabel(r"Level $\epsilon$ of the prediction sets"); a.set_ylabel("Test tasks (%)"); a.set_ylim(0, 100); a.legend(loc="center left", fontsize=5.6); ttl(a, "b", "Risk and coverage by level")
    b1 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "b1_seed*.csv"))]); a = ax[0, 2]
    for xi, col, mk, ls in [(0.05, BLUE, "o", "-"), (0.10, GREEN, "D", "--"), (0.20, YELLOW, "s", "-.")]:
        gg = b1[b1.xi == xi].groupby("n_cal")["eq"].mean(); a.plot(gg.index, 100 * gg.values, color=col, marker=mk, ls=ls, label=rf"$\xi$ = {xi:.2f}", markeredgecolor="white", markeredgewidth=0.4)
    a.set_xticks([45, 100, 200, 300]); a.set_xlabel("Calibration tasks $n_{cal}$"); a.set_ylabel("Tasks equipped (%)"); a.set_ylim(0, 100); a.legend(loc="lower right", fontsize=5.6); ttl(a, "c", "Calibration power")
    x6 = LX("X6"); a = ax[0, 3]; sc = ["matched", "types: low-stakes to assessment", "types: assessment to low-stakes", "domains: programming+language to data science", "domains: data science to programming+language"]
    labs = ["matched", "low$\\to$assess.", "assess.$\\to$low", "P+L$\\to$DS", "DS$\\to$P+L"]; w = 0.36
    for j, (xi, col, hatch) in enumerate([(0.05, BLUE, ""), (0.10, GREEN, "//")]):
        v = [100 * x6[(x6.scenario == s) & (x6.xi == xi)].dfr.mean() for s in sc]
        a.bar(np.arange(5) + (j - 0.5) * w, v, w, color=col, hatch=hatch, edgecolor="white", linewidth=0.3, label=rf"$\xi$ = {xi:.2f}"); a.axhline(100 * xi, color=col, ls=":", lw=0.8)
    a.set_xticks(range(5)); a.set_xticklabels(labs, rotation=28, ha="right", fontsize=5.6); a.set_ylabel("Test denied-flow rate (%)"); a.set_ylim(0, 14); a.legend(loc="upper left", fontsize=5.6); ttl(a, "d", "Distribution shift")
    b2 = add(L("B2")); ms = ["EmerKit", "EmerKit-declared", "RAG-MCP+filter", "Zeng+filter"]
    a = ax[1, 0]; sub = b2[(b2.w == 0.05) & (b2.cue == 0.4)]; mat = np.array([[100 * sub[(sub.method == m) & (sub.m == x)].dfr_e.mean() for x in [0.0, 0.2, 0.4, 0.6, 0.8]] for m in ms])
    heat(a, mat, [M[m][0] for m in ms], ["0", "0.2", "0.4", "0.6", "0.8"], fmt="{:.1f}", cmap="YlOrRd", vmin=0, vmax=22); a.set_xlabel("Share of missing annotations"); ttl(a, "e", "Denied-flow rate (%)")
    a = ax[1, 1]; ln(a, b2[(b2.m == 0.6) & (b2.cue == 0.4)], "w", "dfr_e", ms); a.set_xticks([0, 0.05, 0.1, 0.2]); a.set_xlabel("Share of wrong annotations"); a.set_ylabel("Denied-flow rate (%)"); a.set_ylim(0, 25); ttl(a, "f", "Wrong annotations")
    b3 = add(L("B3")); a = ax[1, 2]; ln(a, b3[b3.profile == "standard"], "ext_bias", "dfr_e", ms); a.set_xscale("log", base=2); a.set_xticks([0.5, 1, 2]); a.set_xticklabels(["0.5", "1", "2"])
    a.set_xlabel("Weight of third-party providers"); a.set_ylabel("Denied-flow rate (%)"); a.set_ylim(0, 25); ttl(a, "g", "Third-party providers")
    a = ax[1, 3]; wf = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "wf_calibration", "wfcal_seed*.csv"))]).groupby("calibration")[["workflow_risk", "all_equipped", "task_equipped"]].mean(); w = 0.36
    for j, (name, col, hatch, lab) in enumerate([("task-level xi=0.05", BLUE, "", "per task"), ("workflow-level xi=0.05", YELLOW, "//", "per workflow")]):
        v = 100 * wf.loc[name, ["workflow_risk", "all_equipped", "task_equipped"]].values; r = a.bar(np.arange(3) + (j - 0.5) * w, v, w, color=col, hatch=hatch, edgecolor="white", linewidth=0.3, label=lab)
        for x, y in zip(r, v): val(a, x.get_x() + w / 2, y, f"{y:.0f}")
    a.set_xticks(range(3)); a.set_xticklabels(["risky\nworkflows", "fully\nequipped", "tasks\nequipped"], fontsize=5.8); a.set_ylabel("Share (%)"); a.set_ylim(0, 138); a.set_yticks([0, 25, 50, 75, 100])
    a.legend(loc="upper left", fontsize=5.6, title="Calibration", title_fontsize=5.6); ttl(a, "h", "Workflow bound")
    toplegend(fig, SIX[:5] + ["Oracle", "EmerKit-declared"]); save(fig, "fig_rq2")

# ============================================================== Figure 3: composition, revision, and request complexity (RQ3)
def fig_rq3():
    fig, ax = new(); c = L("C2"); c = c[c["no_workflow"] != 1] if "no_workflow" in c else c
    a = ax[0, 0]; e = c[(c.method == "EmerKit") & (c.density == "medium") & (c.nu == 0.5) & (c["mode"] == "isolated")].groupby("s_max").all_equipped.mean()
    a.plot(e.index, 100 * e.values, color=BLUE, marker="o", markeredgecolor="white", markeredgewidth=0.4)
    for den, col, mk in [("low", CYAN, "v"), ("high", PURPLE, "^")]:
        v = c[(c.method == "EmerKit") & (c.density == den)].all_equipped.mean(); a.scatter([2], [100 * v], color=col, marker=mk, s=22, zorder=3, label=f"{den}-density catalogue")
    a.set_xticks([0, 1, 2, 3]); a.set_xlabel("Substitutions allowed $s_{\\max}$"); a.set_ylabel("Fully equipped workflows (%)"); a.set_ylim(40, 100); a.legend(loc="lower right", fontsize=5.4); ttl(a, "a", "Effect of substitution")
    a = ax[0, 1]; risk = c[c.config == 0].groupby("method").any_dfr.mean()
    c1 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(LLM, "C1_*seed*.csv"))]); c1 = c1[c1.parsed == 1].groupby("method").any_dfr.mean()
    labs = ["EmerKit", "QoS-IP+F", "RAG-MCP+F", "QoS-IP", "RAG-MCP", "EmerKit-LLM", "CP"]; cols = [BLUE, GREEN, PURPLE, YELLOW, RED, CYAN, INK]
    vals = [100 * risk["EmerKit"], 100 * risk["Gen+Zeng+filter"], 100 * risk["Gen+RAG-MCP+filter"], 100 * risk["Gen+Zeng"], 100 * risk["Gen+RAG-MCP"], 100 * c1["LLM-gen+EmerKit"], 100 * c1["Compositio Prompto"]]
    yp = list(range(7))[::-1]; a.barh(yp, vals, color=cols, height=0.66); a.set_yticks(yp); a.set_yticklabels(labs, fontsize=5.8); a.axhline(1.5, color=INK, lw=0.6, ls=":")
    for y_, v in zip(yp, vals): a.text(v + 2, y_, f"{v:.0f}", va="center", fontsize=5.6)
    a.set_xlim(0, 118); a.set_xticks([0, 25, 50, 75, 100]); a.set_xlabel("Workflows with a risky toolkit (%)"); a.grid(axis="y", visible=False); ttl(a, "b", "End-to-end risk")
    x4 = LX("X4"); wm = ["EmerKit", "EmerKit-nosub", "RAG-MCP+filter", "Zeng", "Zeng+filter"]
    a = ax[0, 2]; ln(a, x4, "L", "all_equipped", wm); a.set_xticks([3, 4, 5, 6, 8, 10]); a.set_xlabel("Workflow length (tasks)"); a.set_ylabel("Fully equipped workflows (%)"); a.set_ylim(-3, 103); ttl(a, "c", "Length: coverage")
    a = ax[0, 3]; ln(a, x4, "L", "any_dfr", wm); a.set_xticks([3, 4, 5, 6, 8, 10]); a.set_xlabel("Workflow length (tasks)"); a.set_ylabel("Workflows with a\nrisky toolkit (%)"); a.set_ylim(-3, 103); ttl(a, "d", "Length: risk")
    x1 = LX("X1"); x1["valid"] = ((x1["eq"] == 1) & (x1["cov"] == 1) & (x1["dfr"] == 0) & (x1["qviol"] == 0)).astype(float); x1["dfr_e"] = np.where(x1["eq"] == 1, x1["dfr"], np.nan)
    x1["tools"] = x1.groupby("config").n_tools.transform("mean").round(-1)
    for a, y, lab, t, let in [(ax[1, 0], "valid", "Valid toolkits (%)", "Catalogue size: validity", "e"), (ax[1, 1], "dfr_e", "Denied-flow rate (%)", "Catalogue size: risk", "f")]:
        ln(a, x1, "tools", y, SIX); a.set_xscale("log"); a.set_xticks([60, 170, 300, 630]); a.set_xticklabels(["60", "170", "300", "630"]); a.minorticks_off()
        a.set_xlabel("Tools in the catalogue"); a.set_ylabel(lab); ttl(a, let, t)
    x2 = LX("X2"); x2["valid"] = ((x2["eq"] == 1) & (x2["cov"] == 1) & (x2["dfr"] == 0) & (x2["qviol"] == 0)).astype(float)
    a = ax[1, 2]; ln(a, x2.assign(nRc=x2.config + 1), "nRc", "valid", SIX); a.set_xticks([1, 2, 3, 4, 5]); a.set_xlabel("Capabilities per task"); a.set_ylabel("Valid toolkits (%)"); a.set_ylim(-3, 100); ttl(a, "g", "Task complexity")
    x5 = LX("X5"); a = ax[1, 3]; g = x5[x5.N == 20].groupby("nR")[["exact_ok", "greedy_ok"]].mean(); w = 0.36
    a.bar(np.arange(len(g)) - w / 2, 100 * g.exact_ok, w, color=BLUE, label="exact"); a.bar(np.arange(len(g)) + w / 2, 100 * g.greedy_ok, w, color=YELLOW, hatch="//", edgecolor="white", linewidth=0.3, label="greedy")
    a.set_xticks(range(len(g))); a.set_xticklabels(g.index); a.set_xlabel("Capabilities per task"); a.set_ylabel("Tasks with a toolkit (%)"); a.set_ylim(0, 100); a.legend(loc="upper right", fontsize=5.6); ttl(a, "h", "Exact or greedy")
    toplegend(fig, ["EmerKit", "EmerKit-nosub", "RAG-MCP", "RAG-MCP+filter", "Zeng", "Zeng+filter", "Oracle"]); save(fig, "fig_rq3")

# ============================================================== Figure 4: scalability, gap map, sensitivity
def fig_rq4():
    fig, ax = new(); a = ax[0, 0]; c3 = L("C3"); g = c3[(c3.nR == 3) & (c3.k_max == 4) & (c3.T_max == 10.0)].groupby("N").time.agg(med="median", p95=lambda x: np.percentile(x, 95))
    a.plot(g.index, 1000 * g.med, color=BLUE, marker="o", label="median"); a.plot(g.index, 1000 * g.p95, color=RED, marker="^", ls="--", label="95th percentile")
    a.set_xticks([10, 30, 50, 100]); a.set_xlabel("Shortlist bound $N$"); a.set_ylabel("Solving time per task (ms)"); a.legend(loc="lower right", fontsize=5.6); ttl(a, "a", "Solver scaling")
    a = ax[0, 1]; x1 = LX("X1"); x1["tools"] = x1.groupby("config").n_tools.transform("mean").round(-1)
    for m in ["EmerKit", "Zeng", "RAG-MCP"]:
        gg = x1[x1.method == m].groupby("tools").time.median(); n, c, mk, ls = M[m]; a.plot(gg.index, 1000 * gg.values, color=c, marker=mk, ls=ls, label=n, markeredgecolor="white", markeredgewidth=0.4)
    c4 = L("C4").groupby("size")[["emerkit_s", "prep_s"]].median(); a.plot(c4.index, 1000 * c4.emerkit_s, color=BLUE, marker="o", ls=":", mfc="white", label="EmerKit, + real tools")
    a.plot(c4.index, 1000 * c4.prep_s, color=INK, marker="x", ls=":", label="preparation (once)")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xticks([60, 300, 1000, 3000]); a.set_xticklabels(["60", "300", "1k", "3k"]); a.minorticks_off(); a.set_xlabel("Tools in the catalogue"); a.set_ylabel("Time (ms)")
    a.set_ylim(0.4, 9000); a.legend(loc="upper left", fontsize=4.8, handlelength=1.5, ncol=2, columnspacing=0.5); ttl(a, "b", "Composition time")
    e1 = L("E1"); sh = (e1.groupby(["catalogue", "profile", "cls"]).dem.sum() / e1.groupby(["catalogue", "profile"]).dem.sum()).unstack().fillna(0); a = ax[0, 2]
    order = [(cg, p) for cg in ["synthetic", "mixed"] for p in ["permissive", "standard", "strict"]]; bottom = np.zeros(6)
    for cls, col, hatch in [("SAFE", GREEN, ""), ("UNCERTAIN", YELLOW, "//"), ("RISKY", RED, "xx")]:
        v = np.array([100 * sh.loc[o, cls] for o in order]); a.bar(range(6), v, bottom=bottom, color=col, hatch=hatch, edgecolor="white", linewidth=0.3, label=cls.capitalize(), width=0.7); bottom += v
    a.set_xticks(range(6)); a.set_xticklabels([f"{'S' if cg == 'synthetic' else 'R'}-{dict(permissive='perm', standard='std', strict='strict')[p]}" for cg, p in order], fontsize=5.6, rotation=28, ha="right")
    a.set_ylim(0, 128); a.set_yticks([0, 25, 50, 75, 100]); a.set_ylabel("Archetypes (%)"); a.legend(loc="upper center", ncol=3, fontsize=5.3, columnspacing=0.6, handlelength=1.0); ttl(a, "c", "Gap map")
    e2 = L("E2"); a = ax[0, 3]; s2 = (e2.groupby(["remove", "annotate", "cls"]).dem.sum() / e2.groupby(["remove", "annotate"]).dem.sum()).unstack().fillna(0); base0 = sh.loc[("synthetic", "standard")]
    for cls, col, mk, ls in [("UNCERTAIN", YELLOW, "s", "-"), ("RISKY", RED, "^", "--")]:
        a.plot([0, 0.25, 0.5], [100 * base0[cls]] + [100 * s2.loc[(0.0, x), cls] for x in [0.25, 0.5]], color=col, marker=mk, ls=ls, label=cls.capitalize())
    a.set_xticks([0, 0.25, 0.5]); a.set_xlabel("Share of tools annotated"); a.set_ylabel("Archetypes (%)"); a.set_ylim(0, 10); a.legend(loc="center right", fontsize=5.6); ttl(a, "d", "Published annotations")
    a = ax[1, 0]; a4 = add(L("A4")); ln(a, a4, "tau", "valid", ["EmerKit", "Zeng", "RAG-MCP"]); a.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5]); a.set_xlabel(r"Tightness $\tau$ of the QoS bounds"); a.set_ylabel("Valid toolkits (%)"); a.set_ylim(-3, 100)
    a.legend(loc="upper left", fontsize=5.6); ttl(a, "e", "QoS bounds")
    a = ax[1, 1]; a1 = A1(); types = ["acquisition", "practice", "formative", "summative", "discussion", "production"]; ms = ["EmerKit", "RAG-MCP+filter", "Zeng", "Zeng+filter", "Oracle"]
    mat = np.array([[100 * a1[(a1.method == m) & (a1.type == t)].valid.mean() for t in types] for m in ms])
    heat(a, mat, [M[m][0] for m in ms], ["acq.", "prac.", "form.", "summ.", "disc.", "prod."], vmin=0, vmax=100); a.tick_params(axis="x", labelsize=4.9); ttl(a, "f", "Valid toolkits (%) by type")
    a = ax[1, 2]; seeds = a1[a1["eq"] == 1].groupby(["method", "seed"]).dfr.mean().reset_index(); order = ["EmerKit", "RAG-MCP+filter", "Zeng", "Zeng+filter"]
    vp = a.violinplot([100 * seeds[seeds.method == m].dfr.values for m in order], showmeans=True, widths=0.75)
    for b_, m in zip(vp["bodies"], order): b_.set_facecolor(M[m][1]); b_.set_alpha(0.75); b_.set_edgecolor(INK); b_.set_linewidth(0.4)
    for k_ in ("cmeans", "cmins", "cmaxes", "cbars"): vp[k_].set_color(INK); vp[k_].set_linewidth(0.6)
    for i, m in enumerate(order): a.scatter(np.full(5, i + 1), 100 * seeds[seeds.method == m].dfr.values, s=5, color=INK, zorder=3)
    a.set_xticks(range(1, 5)); a.set_xticklabels([M[m][0] for m in order], rotation=28, ha="right", fontsize=5.8); a.set_ylabel("DFR per seed (%)"); ttl(a, "g", "Variability across seeds")
    a = ax[1, 3]; d2 = add(L("D2")); g = d2.groupby(["lam_I", "lam_C", "lam_Q", "theta"]).valid.mean(); ys = range(len(g))
    a.hlines(ys, 80, 100 * g.values, color=BLUE, lw=0.9); a.plot(100 * g.values, ys, "o", color=BLUE, ms=3.2); a.set_yticks(ys); a.set_yticklabels([f"{i[0]:g}/{i[1]:g}/{i[2]:g}/{i[3]:g}" for i in g.index], fontsize=5.0)
    a.set_xlim(80, 86); a.set_xlabel("Valid toolkits (%)"); a.set_ylabel(r"$\lambda_I/\lambda_C/\lambda_Q/\theta$", fontsize=6); ttl(a, "h", "Weights")
    toplegend(fig, ["EmerKit", "RAG-MCP", "RAG-MCP+filter", "Zeng", "Zeng+filter", "Oracle"]); save(fig, "fig_rq4")

if __name__ == "__main__":
    for f in (fig_rq1, fig_rq2, fig_rq3, fig_rq4): f()
