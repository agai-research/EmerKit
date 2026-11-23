"""Comparative tables of the paper (best deployable value of each column in bold; the oracle is an upper bound and is excluded)
and macros of the additional series. Writes report/tab_*.tex and report/macros5.tex."""
import os, sys, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_grid import L, add
REP = os.path.join(HERE, "report"); RES = os.path.join(HERE, "results")
def LX(s): return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(RES, "extra", f"{s}_c*.csv")))], ignore_index=True)
NAME = {"EmerKit": "\\sysname{}", "EmerKit-xi0.10": "\\sysname{} ($\\xi{=}0.10$)", "Oracle": "Oracle", "RAG-MCP": "RAG-MCP", "RAG-MCP+filter": "RAG-MCP+F",
        "Zeng": "QoS-IP", "Zeng+filter": "QoS-IP+F", "EmerKit-declared": "\\sysname{}-DA", "EmerKit-nosub": "\\sysname{}-NS",
        "Gen+RAG-MCP": "RAG-MCP", "Gen+RAG-MCP+filter": "RAG-MCP+F", "Gen+Zeng": "QoS-IP", "Gen+Zeng+filter": "QoS-IP+F"}
MAC = {}
def mac(k, v, pct=True, nd=1): MAC[k] = f"{100 * v:.{nd}f}" if pct else (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))

def render(rows, cols, better, fmt, path, header, skip=("Oracle",), align=None, top=None):
    best = {}
    for c in cols:
        vals = [r[1][c] for r in rows if r[0] not in skip and r[0] != "--rule--" and r[1].get(c) is not None and not pd.isna(r[1][c])]
        best[c] = None if better.get(c) is None or not vals else (max(vals) if better[c] == "max" else min(vals))
    out = ["\\begin{tabular}{" + (align or ("l" + "r" * len(cols))) + "}\\toprule"] + ([top] if top else []) + [header + " \\\\ \\midrule"]
    for lab, v in rows:
        if lab == "--rule--": out.append("\\midrule"); continue
        cells = []
        for c in cols:
            x = v.get(c)
            if x is None or pd.isna(x): cells.append("--"); continue
            t = fmt[c].format(x) if isinstance(fmt, dict) else fmt.format(x)
            cells.append(f"\\textbf{{{t}}}" if lab not in skip and best[c] is not None and abs(x - best[c]) < 1e-9 else t)
        out.append(lab + " & " + " & ".join(cells) + " \\\\")
    out.append("\\bottomrule\\end{tabular}"); open(os.path.join(REP, path), "w").write("\n".join(out) + "\n")

P = lambda x: 100 * x
valid = lambda d: ((d["eq"] == 1) & (d["cov"] == 1) & (d["dfr"] == 0) & (d["qviol"] == 0)).astype(float)
SIX = ["EmerKit", "RAG-MCP", "RAG-MCP+filter", "Zeng", "Zeng+filter", "--rule--", "Oracle"]

# ---- T1 reference comparison with validity by pedagogical type
a1 = add(pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "a1_seed*.csv"))]))
types = ["acquisition", "practice", "formative", "summative", "discussion", "production"]
g = a1.groupby("method").agg(eq=("eq", "mean"), valid=("valid", "mean"), dfr=("dfr_e", "mean"), q=("qviol_e", "mean"), red=("red_e", "mean"), load=("lb_e", "mean"))
size = a1[a1["eq"] == 1].groupby("method")["size"].mean(); vt = a1.pivot_table(index="method", columns="type", values="valid")
rows = []
for m in ["EmerKit", "EmerKit-xi0.10"] + SIX[1:]:
    if m == "--rule--": rows.append(("--rule--", {})); continue
    r = dict(eq=P(g.loc[m, "eq"]), valid=P(g.loc[m, "valid"]), dfr=P(g.loc[m, "dfr"]), q=P(g.loc[m, "q"]), red=g.loc[m, "red"], load=P(g.loc[m, "load"]), size=size[m])
    r.update({t: P(vt.loc[m, t]) for t in types}); rows.append((NAME[m], r))
cols = ["eq", "valid", "dfr", "q", "red", "load", "size"] + types
render(rows, cols, {**dict(eq="max", valid="max", dfr="min", q="min", red="min", load="min", size=None), **{t: "max" for t in types}},
       {**dict(eq="{:.1f}", valid="{:.1f}", dfr="{:.1f}", q="{:.1f}", red="{:.2f}", load="{:.1f}", size="{:.2f}"), **{t: "{:.1f}" for t in types}}, "tab_main.tex",
       "Method & Equip. $\\uparrow$ & Valid $\\uparrow$ & DFR $\\downarrow$ & QoS $\\downarrow$ & Red. $\\downarrow$ & Load $\\downarrow$ & Size & Acq. & Prac. & Form. & Summ. & Disc. & Prod.",
       top="& \\multicolumn{7}{c}{All tasks} & \\multicolumn{6}{c}{Valid toolkits by pedagogical type $\\uparrow$} \\\\ \\cmidrule(lr){2-8}\\cmidrule(lr){9-14}")

# ---- T2 catalogue size
x1 = LX("X1"); x1["valid"] = valid(x1); x1["dfr_e"] = np.where(x1["eq"] == 1, x1["dfr"], np.nan); sizes = x1.groupby("config").n_tools.mean().round(-1).astype(int).tolist()
pv, pd_, pt = x1.pivot_table(index="method", columns="config", values="valid"), x1.pivot_table(index="method", columns="config", values="dfr_e"), x1.pivot_table(index="method", columns="config", values="time", aggfunc="median")
rows = [("--rule--", {}) if m == "--rule--" else (NAME[m], {**{f"v{c}": P(pv.loc[m, c]) for c in range(6)}, **{f"d{c}": P(pd_.loc[m, c]) for c in range(6)}, "t": 1000 * pt.loc[m, 5]}) for m in SIX]
cols = [f"v{c}" for c in range(6)] + [f"d{c}" for c in range(6)] + ["t"]
render(rows, cols, {**{f"v{c}": "max" for c in range(6)}, **{f"d{c}": "min" for c in range(6)}, "t": "min"}, "{:.1f}", "tab_scale.tex",
       "Method & " + " & ".join(map(str, sizes)) + " & " + " & ".join(map(str, sizes)) + " & " + str(sizes[-1]),
       top="& \\multicolumn{6}{c}{Valid toolkits (\\%) $\\uparrow$ by catalogue size} & \\multicolumn{6}{c}{DFR (\\%) $\\downarrow$ by catalogue size} & ms \\\\ \\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\\cmidrule(lr){14-14}")
e = x1[x1.method == "EmerKit"].groupby("config"); mac("XoneEmkValidMin", e.valid.mean().min()); mac("XoneEmkValidMax", e.valid.mean().max()); mac("XoneEmkDfrMax", e.dfr_e.mean().max())
mac("XoneZfDfrMax", pd_.loc["Zeng+filter"].max()); mac("XoneZfValidMax", pv.loc["Zeng+filter"].max()); MAC["XoneEmkMs"] = f"{1000 * pt.loc['EmerKit', 5]:.0f}"; MAC["XoneZMs"] = f"{1000 * pt.loc['Zeng', 5]:.0f}"
MAC["XoneMin"], MAC["XoneMax"] = str(sizes[0]), str(sizes[-1])

# ---- T3 capabilities per task
x2 = LX("X2"); x2["valid"] = valid(x2); x2["dfr_e"] = np.where(x2["eq"] == 1, x2["dfr"], np.nan)
pv, pd_ = x2.pivot_table(index="method", columns="config", values="valid"), x2.pivot_table(index="method", columns="config", values="dfr_e")
rows = [("--rule--", {}) if m == "--rule--" else (NAME[m], {**{f"v{c}": P(pv.loc[m, c]) for c in range(5)}, **{f"d{c}": P(pd_.loc[m, c]) for c in range(5)}}) for m in SIX]
cols = [f"v{c}" for c in range(5)] + [f"d{c}" for c in range(5)]
render(rows, cols, {**{f"v{c}": "max" for c in range(5)}, **{f"d{c}": "min" for c in range(5)}}, "{:.1f}", "tab_caps.tex", "Method & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5",
       top="& \\multicolumn{5}{c}{Valid toolkits (\\%) $\\uparrow$} & \\multicolumn{5}{c}{DFR (\\%) $\\downarrow$} \\\\ \\cmidrule(lr){2-6}\\cmidrule(lr){7-11}")
mac("XtwoEmkOne", pv.loc["EmerKit", 0]); mac("XtwoEmkFive", pv.loc["EmerKit", 4]); mac("XtwoOraFive", pv.loc["Oracle", 4]); mac("XtwoZfFive", pv.loc["Zeng+filter", 4])
mac("XtwoZfDfrFive", pd_.loc["Zeng+filter", 4]); mac("XtwoEmkDfrMax", pd_.loc["EmerKit"].max()); mac("XtwoRagFive", pv.loc["RAG-MCP+filter", 4])

# ---- T4 workflow length
x4 = LX("X4"); Ls = [3, 4, 5, 6, 8, 10]; pa, pr = x4.pivot_table(index="method", columns="L", values="all_equipped"), x4.pivot_table(index="method", columns="L", values="any_dfr")
wm = ["EmerKit", "EmerKit-nosub", "RAG-MCP", "RAG-MCP+filter", "Zeng", "Zeng+filter"]
rows = [(NAME[m], {**{f"a{l}": P(pa.loc[m, l]) for l in Ls}, **{f"r{l}": P(pr.loc[m, l]) for l in Ls}}) for m in wm]
cols = [f"a{l}" for l in Ls] + [f"r{l}" for l in Ls]
render(rows, cols, {**{f"a{l}": "max" for l in Ls}, **{f"r{l}": "min" for l in Ls}}, "{:.1f}", "tab_len.tex", "Method & " + " & ".join(map(str, Ls)) + " & " + " & ".join(map(str, Ls)),
       top="& \\multicolumn{6}{c}{Fully equipped workflows (\\%) $\\uparrow$} & \\multicolumn{6}{c}{Workflows with a risky toolkit (\\%) $\\downarrow$} \\\\ \\cmidrule(lr){2-7}\\cmidrule(lr){8-13}")
mac("XfourEmkThree", pa.loc["EmerKit", 3]); mac("XfourEmkTen", pa.loc["EmerKit", 10]); mac("XfourNsTen", pa.loc["EmerKit-nosub", 10]); mac("XfourZfTen", pa.loc["Zeng+filter", 10])
mac("XfourEmkRiskThree", pr.loc["EmerKit", 3]); mac("XfourEmkRiskTen", pr.loc["EmerKit", 10]); mac("XfourZfRiskTen", pr.loc["Zeng+filter", 10]); mac("XfourZRiskTen", pr.loc["Zeng", 10]); mac("XfourZTen", pa.loc["Zeng", 10])

# ---- T5 robustness (denied-flow rate)
b2 = add(L("B2")); b3 = add(L("B3")); sm = ["EmerKit", "EmerKit-declared", "RAG-MCP+filter", "Zeng+filter"]
rows = []
for m in sm:
    r = {f"m{x}": P(b2[(b2.method == m) & (b2.w == 0.05) & (b2.cue == 0.4) & (b2.m == x)].dfr_e.mean()) for x in [0.0, 0.2, 0.4, 0.6, 0.8]}
    r.update({f"w{x}": P(b2[(b2.method == m) & (b2.m == 0.6) & (b2.cue == 0.4) & (b2.w == x)].dfr_e.mean()) for x in [0.0, 0.1, 0.2]})
    r.update({f"e{x}": P(b3[(b3.method == m) & (b3.profile == "standard") & (b3.ext_bias == x)].dfr_e.mean()) for x in [0.5, 1.0, 2.0]})
    r.update({f"p{x}": P(b3[(b3.method == m) & (b3.profile == x)].dfr_e.mean()) for x in ["permissive", "standard", "strict"]}); rows.append((NAME[m], r))
cols = list(rows[0][1])
render(rows, cols, {c: "min" for c in cols}, "{:.1f}", "tab_robust.tex", "Method & 0 & 0.2 & 0.4 & 0.6 & 0.8 & 0 & 0.1 & 0.2 & 0.5 & 1 & 2 & perm. & std & strict",
       top="& \\multicolumn{5}{c}{Missing annotations} & \\multicolumn{3}{c}{Wrong annotations} & \\multicolumn{3}{c}{Third-party weight} & \\multicolumn{3}{c}{Norm profile} \\\\ \\cmidrule(lr){2-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\\cmidrule(lr){13-15}")

# ---- T6 end to end
c = L("C2"); c = c[c["no_workflow"] != 1] if "no_workflow" in c else c
def e2e(df): return dict(eq=P(df.equipped.mean()), full=P(df.all_equipped.mean()), risk=P(df.any_dfr.mean()), obj=P(df.obj_cov.mean()))
rows = [("\\sysname{}-NS", e2e(c[(c.config == 1) & (c.method == "EmerKit")])), ("\\sysname{}", e2e(c[(c.config == 0) & (c.method == "EmerKit")]))]
rows += [(NAME[m], e2e(c[(c.config == 0) & (c.method == m)])) for m in ["Gen+RAG-MCP", "Gen+RAG-MCP+filter", "Gen+Zeng", "Gen+Zeng+filter"]]
c1 = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "llm", "C1_*seed*.csv"))]); v = c1[c1.parsed == 1]; rows.append(("--rule--", {}))
for m, lab in [("LLM-gen+EmerKit", "\\sysname{}-LLM"), ("Compositio Prompto", "CP")]:
    d = v[v.method == m]; rows.append((lab, dict(eq=P(d.equipped.mean()), full=None, risk=P(d.any_dfr.mean()), obj=P(d.obj_cov.mean()))))
render(rows, ["eq", "full", "risk", "obj"], dict(eq="max", full="max", risk="min", obj="max"), "{:.1f}", "tab_e2e.tex", "Method & Tasks eq. $\\uparrow$ & Fully eq. $\\uparrow$ & Risky $\\downarrow$ & Obj. $\\uparrow$", skip=())

# ---- T7 ablations
d1 = add(pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "d1_seed*.csv"))]).rename(columns={"variant": "method"}))
gd = d1.groupby("method").agg(eq=("eq", "mean"), valid=("valid", "mean"), dfr=("dfr_e", "mean"), q=("qviol_e", "mean"), red=("red_e", "mean"), load=("lb_e", "mean"))
rows = [("\\sysname{} (full)", dict(eq=P(g.loc["EmerKit", "eq"]), valid=P(g.loc["EmerKit", "valid"]), dfr=P(g.loc["EmerKit", "dfr"]), q=P(g.loc["EmerKit", "q"]), red=g.loc["EmerKit", "red"], load=P(g.loc["EmerKit", "load"])))]
lab = {"additive utility": "additive utility", "declared annotations": "declared annotations (DA)", "no QoS": "no QoS", "no interactions": "no interactions (NI)"}
rows += [(lab.get(m, m), dict(eq=P(gd.loc[m, "eq"]), valid=P(gd.loc[m, "valid"]), dfr=P(gd.loc[m, "dfr"]), q=P(gd.loc[m, "q"]), red=gd.loc[m, "red"], load=P(gd.loc[m, "load"]))) for m in gd.index]
render(rows, ["eq", "valid", "dfr", "q", "red", "load"], dict(eq="max", valid="max", dfr="min", q="min", red="min", load="min"),
       dict(eq="{:.1f}", valid="{:.1f}", dfr="{:.1f}", q="{:.1f}", red="{:.2f}", load="{:.1f}"), "tab_abl.tex", "Variant & Eq. $\\uparrow$ & Valid $\\uparrow$ & DFR $\\downarrow$ & QoS $\\downarrow$ & Red. $\\downarrow$ & Load $\\downarrow$", skip=())

# ---- T8 exact vs greedy
x5 = LX("X5"); both = x5[(x5.exact_ok == 1) & (x5.greedy_ok == 1)].copy(); both["ratio"] = both.greedy_obj / both.exact_obj
gg = x5.groupby(["N", "nR"]).agg(ex=("exact_ok", "mean"), gr=("greedy_ok", "mean"), exms=("exact_s", lambda v_: 1000 * v_.median()), grms=("greedy_s", lambda v_: 1000 * v_.median()))
gg["ratio"] = both.groupby(["N", "nR"]).ratio.mean(); gg["opt"] = both.assign(o=both.ratio > 0.999).groupby(["N", "nR"]).o.mean()
out = ["\\begin{tabular}{rrrrrrrr}\\toprule", "& & \\multicolumn{2}{c}{Tasks equipped (\\%)} & \\multicolumn{2}{c}{Greedy objective} & \\multicolumn{2}{c}{Median time (ms)} \\\\ \\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
       "$N$ & $|R|$ & exact & greedy & ratio & optimal (\\%) & exact & greedy \\\\ \\midrule"]
for (N, nR), r in gg.iterrows():
    out.append(f"{N} & {nR} & \\textbf{{{P(r.ex):.1f}}} & {P(r.gr):.1f} & {r.ratio:.3f} & {P(r.opt):.1f} & {r.exms:.1f} & \\textbf{{{r.grms:.1f}}} \\\\")
out.append("\\bottomrule\\end{tabular}"); open(os.path.join(REP, "tab_greedy.tex"), "w").write("\n".join(out) + "\n")
mac("XfiveLoss", (x5.exact_ok - x5.greedy_ok).groupby([x5.N, x5.nR]).mean().max()); MAC["XfiveRatio"] = f"{both.ratio.mean():.3f}"; MAC["XfiveExMs"] = f"{gg.exms.max():.0f}"; MAC["XfiveGrMs"] = f"{gg.grms.max():.1f}"
mac("XfiveExFifty", gg.loc[(50, 3), "ex"]); mac("XfiveGrFifty", gg.loc[(50, 3), "gr"])

# ---- T9 distribution shift
x6 = LX("X6"); sc = [("matched", "Matched (all types, all domains)"), ("types: low-stakes to assessment", "Low-stakes types $\\to$ assessment types"), ("types: assessment to low-stakes", "Assessment types $\\to$ low-stakes types"),
                     ("domains: programming+language to data science", "Programming, language $\\to$ data science"), ("domains: data science to programming+language", "Data science $\\to$ programming, language")]
out = ["\\begin{tabular}{lrrrrrr}\\toprule", "& \\multicolumn{3}{c}{$\\xi=0.05$} & \\multicolumn{3}{c}{$\\xi=0.10$} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
       "Calibration $\\to$ test & $\\hat\\epsilon$ & DFR & Equip. & $\\hat\\epsilon$ & DFR & Equip. \\\\ \\midrule"]
for s, labl in sc:
    cells = []
    for xi in (0.05, 0.10):
        d = x6[(x6.scenario == s) & (x6.xi == xi)]; dfr = P(d.dfr.mean()); t = f"{dfr:.1f}"
        cells += [f"{d.eps_hat.mean():.3f}", (f"\\underline{{{t}}}" if dfr > 100 * xi else t), f"{P(d.equipped.mean()):.1f}"]
    out.append(labl + " & " + " & ".join(cells) + " \\\\")
out.append("\\bottomrule\\end{tabular}"); open(os.path.join(REP, "tab_shift.tex"), "w").write("\n".join(out) + "\n")
q = lambda s, xi, col: x6[(x6.scenario == s) & (x6.xi == xi)][col].mean()
mac("XsixMatchDfr", q("matched", 0.05, "dfr")); mac("XsixMatchEq", q("matched", 0.05, "equipped")); mac("XsixAssessDfr", q("types: assessment to low-stakes", 0.05, "dfr"))
mac("XsixLowEq", q("types: low-stakes to assessment", 0.05, "equipped")); mac("XsixLowDfr", q("types: low-stakes to assessment", 0.05, "dfr"))
mac("XsixDomMax", max(q(s, 0.05, "dfr") for s, _ in sc[3:]))
open(os.path.join(REP, "macros5.tex"), "w").write("% generated by make_paper_tables.py\n" + "".join(f"\\newcommand{{\\R{k}}}{{{v}}}\n" for k, v in sorted(MAC.items())))
print(MAC)
