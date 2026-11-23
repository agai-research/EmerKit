"""Second part of the report: grid series. Writes report/macros2.tex and report/tables2.tex (generated; do not edit)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_grid import L, add
REP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report")
M = {}; TAB = []
def mac(k, v, pct=True, nd=1):
    M[k] = f"{100 * v:.{nd}f}" if pct else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))
def table(df, caption, label, fmt):
    cols = list(df.columns); idx = list(df.index.names)
    TAB.append("\\begin{table}[t]\\centering\\caption{" + caption + "}\\label{" + label + "}\\scriptsize")
    TAB.append("\\begin{tabular}{" + "l" * len(idx) + "r" * len(cols) + "}\\toprule")
    TAB.append(" & ".join([str(i).replace("_", " ") for i in idx] + [str(c).replace("_", " ") for c in cols]) + " \\\\ \\midrule")
    for key, row in df.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        TAB.append(" & ".join([str(k) for k in key] + [fmt(v) for v in row.values]) + " \\\\")
    TAB.append("\\bottomrule\\end{tabular}\\end{table}")
pc = lambda v: "--" if pd.isna(v) else f"{100 * v:.1f}"

# A4
d = add(L("A4")); t = d.pivot_table(index="tau", columns="method", values="valid")[["EmerKit", "RAG-MCP", "Zeng"]]
table(t, "Series A4: share of valid toolkits (\\%) as a function of the tightness $\\tau$ of the QoS bounds.", "tab:a4", pc)
mac("AfourEmk", t.loc[1.25, "EmerKit"]); mac("AfourZeng", t.loc[1.25, "Zeng"]); mac("AfourRag", t.loc[1.25, "RAG-MCP"])
mac("AfourEmkSevenFive", t.loc[0.75, "EmerKit"]); mac("AfourZengSevenFive", t.loc[0.75, "Zeng"])
# B2
d = add(L("B2")); t = d.pivot_table(index=["m", "w", "cue"], columns="method", values="dfr_e")[["EmerKit", "EmerKit-declared", "RAG-MCP+filter", "Zeng+filter"]]
e = d.pivot_table(index=["m", "w", "cue"], columns="method", values="eq")[["EmerKit", "Oracle", "Zeng+filter"]].add_suffix(" (eq.)")
table(pd.concat([t, e], axis=1), "Series B2: denied-flow rate over equipped tasks and share of equipped tasks (\\%) under missing ($m$) and wrong ($w$) annotations and cue strength; level fixed at $\\epsilon=0.05$.", "tab:b2", pc)
mac("BtwoEmkMin", t["EmerKit"].min()); mac("BtwoEmkMax", t["EmerKit"].max())
mac("BtwoDeclMzero", t.loc[(0.0, 0.05, 0.4), "EmerKit-declared"]); mac("BtwoDeclMeight", t.loc[(0.8, 0.05, 0.4), "EmerKit-declared"])
mac("BtwoZengMax", t["Zeng+filter"].max()); mac("BtwoRagMax", t["RAG-MCP+filter"].max())
mac("BtwoEqWzero", e.loc[(0.6, 0.0, 0.4), "EmerKit (eq.)"]); mac("BtwoEqWtwenty", e.loc[(0.6, 0.2, 0.4), "EmerKit (eq.)"])
# B3
d = add(L("B3")); t = d.pivot_table(index=["profile", "ext_bias"], columns="method", values="dfr_e")[["EmerKit", "EmerKit-declared", "RAG-MCP+filter", "Zeng+filter"]]
table(t, "Series B3: denied-flow rate over equipped tasks (\\%) by norm profile and share of third-party providers (bias).", "tab:b3", pc)
mac("BthreeEmkMax", t["EmerKit"].max()); mac("BthreeZengHigh", t.xs(2.0, level="ext_bias")["Zeng+filter"].max()); mac("BthreeDeclMax", t["EmerKit-declared"].max())
# D2
d = add(L("D2")); t = d.groupby(["lam_I", "lam_C", "lam_Q", "theta"])[["eq", "valid", "dfr_e", "red_e", "over_e"]].mean()
table(t, "Series D2: sensitivity of EmerKit to its weights (one factor at a time; rates in \\%, redundancy per task).", "tab:d2",
      lambda v: f"{v:.3f}" if v < 0.2 and v != 0 and False else pc(v))
mac("DtwoValidMin", t["valid"].min()); mac("DtwoValidMax", t["valid"].max())
mac("DtwoRedNoI", t.loc[(0.0, 1.0, 0.5, 0.75), "red_e"], False, 2); mac("DtwoRedI", t.loc[(1.0, 1.0, 0.5, 0.75), "red_e"], False, 2)
# C2 (requests without a valid workflow excluded for every method)
c = L("C2"); c["no_workflow"] = c["no_workflow"].fillna(0) if "no_workflow" in c else 0
mac("CtwoNoWf", c[(c.method == "EmerKit") & (c.config == 0)].no_workflow.mean())
c = c[c.no_workflow != 1]
g = c.groupby(["config", "s_max", "density", "nu", "mode", "method"]).agg(equipped=("equipped", "mean"), substituted=("substituted", "mean"),
    all_equipped=("all_equipped", "mean"), task_DFR=("dfr", "mean"), workflow_risk=("any_dfr", "mean"), objectives=("obj_cov", "mean"))
emk_all = c[c.method == "EmerKit"]
mac("CtwoObjMin", emk_all.groupby("config").obj_cov.mean().min()); M["CtwoPrereq"] = str(int(emk_all.prereq_viol.sum())); M["CtwoPres"] = str(int(emk_all.preservation_viol.sum()))
g.index = g.index.droplevel(0)
table(g, "Series C2 and C1 (without language models): end-to-end results per request (\\%). Baselines use the same generator, without substitution.", "tab:c2", pc)
emk = c[c.method == "EmerKit"].groupby("config"); ge = emk.all_equipped.mean(); gr = emk.any_dfr.mean(); gd = emk.dfr.mean(); gs = emk.substituted.mean()
mac("CtwoAllZero", ge[1]); mac("CtwoAllTwo", ge[0]); mac("CtwoAllThree", ge[3]); mac("CtwoSub", gs[0]); mac("CtwoTaskDfr", gd[0]); mac("CtwoWfRisk", gr[0])
mac("CtwoWfRiskCons", gr[8]); mac("CtwoAllCons", ge[8])
b = c[c.config == 0].groupby("method")
mac("CtwoZfAll", b.all_equipped.mean()["Gen+Zeng+filter"]); mac("CtwoZfRisk", b.any_dfr.mean()["Gen+Zeng+filter"]); mac("CtwoZRisk", b.any_dfr.mean()["Gen+Zeng"])
mac("CtwoRagRisk", b.any_dfr.mean()["Gen+RAG-MCP"]); mac("CtwoZAll", b.all_equipped.mean()["Gen+Zeng"])
# C3
c = L("C3"); c["infeasible"] = (c.status == "infeasible").astype(float); c["timeout"] = (c.status == "timeout").astype(float)
t = c.groupby(["N", "nR", "k_max", "T_max"]).agg(median_s=("time", "median"), p95_s=("time", lambda x: np.percentile(x, 95)), timeouts=("timeout", "mean"),
    infeasible=("infeasible", "mean"), conflict_groups=("conflict_size", "mean"))
table(t, "Series C3: solver time per task (s), timeouts and infeasible tasks (\\%), and size of the minimal conflict (constraint groups).", "tab:c3",
      lambda v: f"{v:.3f}")
mac("CthreeMedMax", t.median_s.max(), False, 2); mac("CthreePmax", t.p95_s.max(), False, 2); mac("CthreeTo", t.timeouts.max())
mac("CthreeInfFive", t.loc[(20, 5, 4, 10.0), "infeasible"]); mac("CthreeConf", c.conflict_size.mean(), False, 1)
# C4
c = L("C4"); t = c.groupby("size").agg(tools=("n_tools", "mean"), prep_s=("prep_s", "mean"), emerkit_med_s=("emerkit_s", "median"),
    rag_med_s=("rag_s", "median"), zeng_med_s=("zeng_s", "median"), memory_MB=("maxrss_mb", "max"), real_tools_used=("real_in_K", "mean"))
table(t, "Series C4: preparation time, median time per task (s), memory, and share of toolkits that include a real MCP-tools server.", "tab:c4",
      lambda v: f"{v:.4f}" if v < 1 else f"{v:.0f}")
mac("CfourPrep", t.prep_s.max(), False, 2); mac("CfourEmkSmall", 1000 * t.emerkit_med_s.iloc[0], False, 0); mac("CfourEmkLarge", 1000 * t.emerkit_med_s.iloc[-1], False, 0)
mac("CfourMem", t.memory_MB.max(), False, 0); mac("CfourReal", t.real_tools_used.max())
# E1, E2
for s, keys in [("E1", ["catalogue", "profile"]), ("E2", ["remove", "annotate"])]:
    e = L(s); w = (e.groupby(keys + ["cls"]).dem.sum() / e.groupby(keys).dem.sum()).unstack().fillna(0)
    table(w, f"Series {s}: demand-weighted share of task archetypes in each class of the gap map (\\%).", f"tab:{s.lower()}", pc)
    if s == "E1":
        mac("EoneSafeSyn", w.loc[("synthetic", "standard"), "SAFE"]); mac("EoneUncSyn", w.loc[("synthetic", "standard"), "UNCERTAIN"])
        mac("EoneUncMix", w.loc[("mixed", "standard"), "UNCERTAIN"]); mac("EoneRiskStrict", w.loc[("synthetic", "strict"), "RISKY"])
        mac("EoneRiskStd", w.loc[("synthetic", "standard"), "RISKY"])
    else:
        mac("EtwoUncHalf", w.loc[(0.0, 0.5), "UNCERTAIN"])
# F1 (live enactment) and audit
import glob, json, csv
f1 = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(REP, "..", "results", "f1", "F1_exhaustive_seed*.csv"))])
M["FoneN"] = str(len(f1)); M["FoneSeeds"] = str(f1.seed.nunique()); mac("FoneExact", f1.exact_match.mean()); mac("FoneExposed", f1.exposed_only_K.mean())
mac("FoneBlocked", f1.call_outside_blocked.mean()); mac("FoneDenied", f1.denied_observed.gt(0).mean()); M["FoneSec"] = f"{f1.seconds.mean():.1f}"
ev = [json.loads(l) for l in open(os.path.join(REP, "..", "data", "audit", "static_evidence.jsonl")) if l.strip()]
M["AuServers"] = str(sum(1 for r in ev if r.get("fetched"))); M["AuTools"] = str(sum(len(r["tools"]) for r in ev if r.get("fetched")))
rv = list(csv.DictReader(open(os.path.join(REP, "..", "data", "audit", "audit_check_claude.csv"))))
M["AuReviewed"] = str(len(rv)); M["AuOut"] = str(sum(int(r["out"]) for r in rv)); M["AuAgree"] = str(sum(int(r["out"]) == int(r["static_out"]) for r in rv))
M["AuLowConf"] = str(sum(r["confidence"] != "high" for r in rv))
with open(os.path.join(REP, "macros2.tex"), "w") as f:
    f.write("% generated by make_report2.py\n" + "".join(f"\\newcommand{{\\R{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
with open(os.path.join(REP, "tables2.tex"), "w") as f:
    f.write("% generated by make_report2.py\n" + "\n".join(TAB) + "\n")
print(len(M), "macros"); print({k: M[k] for k in sorted(M)})
