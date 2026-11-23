"""Builds report/macros.tex, report/tables.tex and report/fig_*.pdf from the result CSVs (no number is typed by hand)."""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from aggregate import load, boot

HERE = os.path.dirname(__file__); REP = os.path.join(HERE, "report"); os.makedirs(REP, exist_ok=True)
NAVY, TEAL, CHAR, SAND, ROSE, SLATE = "#1f3b5c", "#2a7f7a", "#3a3a3a", "#c9a96e", "#b0645f", "#7a8ca3"
COL = {"EmerKit": NAVY, "EmerKit-xi0.10": TEAL, "Oracle": SLATE, "RAG-MCP": ROSE, "RAG-MCP+filter": SAND, "Zeng": TEAL, "Zeng+filter": CHAR,
       "EmerKit-no-interactions": SLATE}

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": CHAR, "axes.linewidth": 0.6})
macros = {}
def mac(name, val, fmt="{:.1f}", pct=True):
    macros[name] = fmt.format(100 * val if pct else val)
def name_ok(s):
    return "".join(ch for ch in s.title() if ch.isalpha())

# ---------------------------------------------------------------- A1
d = load("a1")
d["valid"] = ((d["eq"] == 1) & (d["cov"] == 1) & (d["dfr"] == 0) & (d["qviol"] == 0)).astype(float)
for k in ["dfr", "qviol", "red", "lb"]:
    d[k + "_eq"] = np.where(d["eq"] == 1, d[k], np.nan)
order = ["EmerKit", "EmerKit-xi0.10", "Oracle", "RAG-MCP", "RAG-MCP+filter", "Zeng", "Zeng+filter"]
rows = []
for m in order:
    g = d[d.method == m]; r = {"m": m}
    for k in ["eq", "valid", "dfr_eq", "qviol_eq", "red_eq", "size"]:
        r[k] = boot(g[k])
    rows.append(r)
    tag = name_ok(m.replace("+", " plus ").replace("-", " "))
    mac(f"A{tag}Eq", r["eq"][0]); mac(f"A{tag}Valid", r["valid"][0]); mac(f"A{tag}Dfr", r["dfr_eq"][0])
    mac(f"A{tag}Qviol", r["qviol_eq"][0]); mac(f"A{tag}Red", r["red_eq"][0], "{:.2f}", False)
mac("NTasks", len(d[d.method == "EmerKit"]), "{:d}", False); macros["NTasks"] = str(len(d[d.method == "EmerKit"]))
macros["NSeeds"] = str(d.seed.nunique())
def cell(t, pct=True):
    mu, lo, hi = t
    return (f"{100*mu:.1f} [{100*lo:.1f}, {100*hi:.1f}]" if pct else f"{mu:.2f}")
tab = ["\\begin{table}[t]\\centering\\caption{Series A1 (pilot): reference comparison on the synthetic catalogue, "
       f"{macros['NTasks']} tasks over {macros['NSeeds']} seeds. Rates in \\%, with 95\\% bootstrap intervals; DFR, QoS violation and "
       "redundancy are computed over equipped tasks.}\\label{tab:a1}", "\\scriptsize\\begin{tabular}{lrrrrr}\\toprule",
       "Method & Equipped & Valid & DFR & QoS viol. & Redund. \\\\ \\midrule"]
for r in rows:
    tab.append(f"{r['m']} & {cell(r['eq'])} & {cell(r['valid'])} & {100*r['dfr_eq'][0]:.1f} & {100*r['qviol_eq'][0]:.1f} & {r['red_eq'][0]:.2f} \\\\")
tab += ["\\bottomrule\\end{tabular}\\end{table}"]

fig, ax = plt.subplots(figsize=(3.4, 2.4))
for r in rows:
    ax.errorbar(100 * r["dfr_eq"][0], 100 * r["eq"][0], xerr=[[100 * (r["dfr_eq"][0] - r["dfr_eq"][1])], [100 * (r["dfr_eq"][2] - r["dfr_eq"][0])]],
                fmt="o", color=COL[r["m"]], ms=4, lw=0.8, capsize=2)
    off = {"RAG-MCP+filter": (-10, -11), "Zeng": (6, -9), "Zeng+filter": (6, -3), "Oracle": (6, 2), "EmerKit": (6, 2), "EmerKit-xi0.10": (6, -8), "RAG-MCP": (-34, -11)}[r["m"]]
    ax.annotate(r["m"], (100 * r["dfr_eq"][0], 100 * r["eq"][0]), textcoords="offset points", xytext=off, fontsize=6.5, color=CHAR)
ax.set_xlabel("Toolkits with a denied flow (%)"); ax.set_ylabel("Tasks equipped (%)"); ax.set_xlim(-3, 85)
fig.tight_layout(); fig.savefig(os.path.join(REP, "fig_a1_tradeoff.pdf")); plt.close(fig)

# ---------------------------------------------------------------- B1
b = load("b1"); t = load("b1_trace")
tr = t.groupby("eps")[["test_eq", "test_dfr", "ambiguous"]].mean()
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.plot(tr.index, 100 * tr.test_dfr, color=ROSE, lw=1.2, marker="o", ms=2, label="Denied-flow rate (test)")
ax.plot(tr.index, 100 * tr.test_eq, color=NAVY, lw=1.2, marker="o", ms=2, label="Tasks equipped (test)")
ax.set_xscale("symlog", linthresh=0.01)
for xi in [0.05, 0.10]:
    ax.axhline(100 * xi, color=CHAR, lw=0.5, ls=":")
    ax.text(0.505, 100 * xi + 1, f"$\\xi={xi:.2f}$", fontsize=6, color=CHAR, ha="right")
ax.set_xlabel("Level $\\epsilon$ of the conformal sets"); ax.set_ylabel("%"); ax.legend(frameon=False, fontsize=6.5, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(REP, "fig_b1_calibration.pdf")); plt.close(fig)
bb = b.groupby(["xi", "n_cal"]).agg(eps=("eps_hat", "mean"), dfr=("dfr", "mean"), eq=("eq", "mean")).reset_index()
tab += ["\\begin{table}[t]\\centering\\caption{Series B1 (pilot): calibrated level $\\hat\\epsilon$ and test denied-flow rate "
        "(over equipped tasks) for each target risk $\\xi$ and number of calibration tasks, $\\zeta=0.1$; means over seeds.}\\label{tab:b1}",
        "\\scriptsize\\begin{tabular}{rrrrr}\\toprule $\\xi$ & $n_{cal}$ & $\\hat\\epsilon$ & DFR (\\%) & Equipped (\\%) \\\\ \\midrule"]
for _, r in bb.iterrows():
    tab.append(f"{r.xi:.2f} & {int(r.n_cal)} & {r.eps:.3f} & {100*r.dfr:.1f} & {100*r["eq"]:.1f} \\\\")
tab += ["\\bottomrule\\end{tabular}\\end{table}"]
from scipy.stats import binom
def tolerated(n, xi, zeta=0.10):
    v = -1
    while binom.cdf(v + 1, n, xi) <= zeta: v += 1
    return v
ns = [45, 100, 200, 300, 500]
tab += ["\\begin{table}[t]\\centering\\caption{Sample-size rule of the task-level calibration: largest number of violations among $n_{cal}$ audited "
        "calibration tasks that still certifies a level, for $\\zeta=0.1$ (computed from the binomial distribution).}\\label{tab:power}",
        "\\scriptsize\\begin{tabular}{l" + "r" * len(ns) + "}\\toprule $n_{cal}$ & " + " & ".join(map(str, ns)) + " \\\\ \\midrule"]
for xi in (0.05, 0.10):
    tab.append(f"$\\xi={xi:.2f}$ & " + " & ".join(str(tolerated(n, xi)) for n in ns) + " \\\\")
tab += ["\\bottomrule\\end{tabular}\\end{table}"]
mac("BMaxDfrRatio", (bb.dfr / bb.xi).max(), "{:.2f}", False)
mac("BEqLow", bb["eq"].min()); mac("BEqHigh", bb["eq"].max())
mac("BEqEpsZero", tr.test_eq.iloc[0]); mac("BAmbEpsZeroFive", tr.ambiguous.loc[0.05])
eh = bb[(bb.xi == 0.05) & (bb.n_cal == 300)].iloc[0]; mac("BEpsThree", eh.eps, "{:.3f}", False); mac("BEqThree", eh["eq"]); mac("BDfrThree", eh.dfr)
eh = bb[(bb.xi == 0.05) & (bb.n_cal == 100)].iloc[0]; mac("BEqHundred", eh["eq"])

# ---------------------------------------------------------------- D1
x = load("d1")
x["valid"] = ((x["eq"] == 1) & (x["cov"] == 1) & (x["dfr"] == 0) & (x["qviol"] == 0)).astype(float)
x["dfr_eq"] = np.where(x["eq"] == 1, x["dfr"], np.nan)
full = d[d.method == "EmerKit"].assign(variant="full EmerKit")
x = pd.concat([full[x.columns.intersection(full.columns)], x], ignore_index=True)
x["qv_eq"] = np.where(x["eq"] == 1, x["qviol"], np.nan); x["red_eq"] = np.where(x["eq"] == 1, x["red"], np.nan)
dd = x.groupby("variant", sort=False).agg(eq=("eq", "mean"), valid=("valid", "mean"), dfr=("dfr_eq", "mean"), qv=("qv_eq", "mean"), red=("red_eq", "mean"))
tab += ["\\begin{table}[t]\\centering\\caption{Series D1 (pilot): ablations at the calibrated level. Rates in \\%; DFR, QoS violation and redundancy over equipped tasks.}\\label{tab:d1}",
        "\\scriptsize\\begin{tabular}{lrrrrr}\\toprule Variant & Equipped & Valid & DFR & QoS viol. & Redund. \\\\ \\midrule"]
for v, r in dd.iterrows():
    tab.append(f"{v} & {100*r["eq"]:.1f} & {100*r.valid:.1f} & {100*r.dfr:.1f} & {100*r.qv:.1f} & {r.red:.2f} \\\\")
    tag = name_ok(v.replace("-", " "))
    mac(f"D{tag}Eq", r["eq"]); mac(f"D{tag}Dfr", r.dfr); mac(f"D{tag}Red", r.red, "{:.2f}", False); mac(f"D{tag}Qviol", r.qv)
tab += ["\\bottomrule\\end{tabular}\\end{table}"]

# ---------------------------------------------------------------- A0 / B0
a = load("a0b0"); am = a.groupby("metric").value.mean()
mac("CapPrec", am["precision"]); mac("CapRec", am["recall"])
mac("LabAccMin", min(am[k] for k in am.index if k.startswith("acc_"))); mac("LabAccMax", max(am[k] for k in am.index if k.startswith("acc_")))
for e, n in [(0.05, "Five"), (0.1, "Ten"), (0.2, "Twenty")]:
    mac(f"ConfCov{n}", am[f"coverage_eps{e}"])

# ---------------------------------------------------------------- A2 / A3
g2 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results/grid/A2_c*.csv"))])
g2["cp1"] = (g2.cp > 0).astype(float)
p2 = g2.pivot_table(index="rho_conf", columns="method", values="cp1")
mac("ARagCpMax", p2["RAG-MCP"].max()); mac("AEmkNoIntCpMax", p2["EmerKit-no-interactions"].max()); mac("AEmkCpMax", p2["EmerKit"].max())
g3 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(HERE, "results/grid/A3_c*.csv"))])
g3["over_eq"] = np.where(g3["eq"] == 1, g3["over"], np.nan); g3["exceed"] = np.where(g3["eq"] == 1, (g3.lb > 1).astype(float), np.nan)
fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.2), sharey=True)
for ax, lf in zip(axs, [1, 4]):
    s = g3[g3.length_factor == lf].pivot_table(index="B", columns="method", values="over_eq")
    for m in ["EmerKit", "EmerKit-no-interactions", "RAG-MCP", "Zeng"]:
        ax.plot(s.index, 100 * s[m], marker="o", ms=3, lw=1, color=COL[m], ls="--" if m.endswith("interactions") else "-", label=m)
    ax.set_xscale("log", base=2); ax.set_xticks(list(s.index)); ax.set_xticklabels([str(v) for v in s.index])
    ax.set_title(f"Definition length $\\times{lf}$", fontsize=7.5); ax.set_xlabel("Context budget $B$ (tokens)")
axs[0].set_ylabel("Toolkits above $\\theta B$ (%)"); axs[1].legend(frameon=False, fontsize=6)
fig.tight_layout(); fig.savefig(os.path.join(REP, "fig_a3_context.pdf")); plt.close(fig)
s1 = g3[(g3.length_factor == 1) & (g3.B == 1000)].groupby("method").over_eq.mean()
mac("AOverEmk", s1["EmerKit"]); mac("AOverNoInt", s1["EmerKit-no-interactions"]); mac("AOverRag", s1["RAG-MCP"]); mac("AOverZeng", s1["Zeng"])
ex = g3[g3.B == 500].groupby("method").exceed.mean(); mac("ARagExceed", ex["RAG-MCP"]); mac("AZengExceed", ex["Zeng"])

with open(os.path.join(REP, "macros.tex"), "w") as f:
    f.write("% generated by make_report.py -- do not edit\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\R{k}}}{{{v}}}\n")
with open(os.path.join(REP, "tables.tex"), "w") as f:
    f.write("% generated by make_report.py -- do not edit\n" + "\n".join(tab) + "\n")
print(len(macros), "macros;", ", ".join(sorted(macros)[:60]))
