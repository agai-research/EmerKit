"""Third part of the report: model-dependent series and panel sensitivity. Writes report/macros3.tex and report/tables3.tex."""
import os, sys, glob, json, csv
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from analyze_grid import L, add
LLM = os.path.join(HERE, "results", "llm"); REP = os.path.join(HERE, "report")
M, TAB = {}, []
def mac(k, v, pct=True, nd=1):
    M[k] = f"{100 * v:.{nd}f}" if pct else (f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v))

# A0 with an LLM extractor
R = json.load(open(glob.glob(os.path.join(LLM, "A0_*seed0.json"))[0]))
def pr(key):
    tp = sum(len(set(r[key]) & set(r["true_caps"])) for r in R); fp = sum(len(set(r[key]) - set(r["true_caps"])) for r in R)
    fn = sum(len(set(r["true_caps"]) - set(r[key])) for r in R); return tp / max(1, tp + fp), tp / max(1, tp + fn)
mac("LaPrec", pr("llm_caps")[0]); mac("LaRec", pr("llm_caps")[1]); mac("LaMapRec", pr("mapper_caps")[1])

# expert panel
J = json.load(open(os.path.join(LLM, "judges_summary.json")))
mac("JnAgree", J["norm"]["majority_agrees_with_current"]); M["JnKappa"] = f"{J['norm']['fleiss_kappa']:.2f}"; M["JnItems"] = str(J["norm"]["items"])
mac("JsAgree", J["sub"]["majority_agrees_with_current"]); M["JsKappa"] = f"{J['sub']['fleiss_kappa']:.2f}"; M["JsItems"] = str(J["sub"]["items"])
M["JfCorr"] = f"{J['fit']['corr_with_current']:.2f}"; M["JfInter"] = f"{J['fit']['inter_model_corr']:.2f}"; M["JfGap"] = f"{J['fit']['mean_abs_gap']:.2f}"
M["JnDis"] = str(len(J["norm"]["disagreements"])); M["JsDis"] = str(len(J["sub"]["disagreements"]))

# A5 on synthetic definitions (ceiling) and on real definitions
s5 = pd.read_csv(os.path.join(LLM, "A5_claude-haiku-5-5_seed0.csv")); mac("AfiveSyn", s5.correct.mean())
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(LLM, "A5real_*.csv"))]); d["log_load"] = np.log(d.load)
M["AfiveN"] = str(len(d)); M["AfiveModels"] = str(d.model.nunique())
acc = d.groupby(["model", "confusable"]).correct.mean()
for m, tag in [("claude-haiku-5-5", "Haiku"), ("claude-opus-5-5", "Opus")]:
    mac(f"Afive{tag}Rand", acc[(m, 0)]); mac(f"Afive{tag}Conf", acc[(m, 1)])
fit, test = d[d.split == "fit"], d[d.split == "test"]
r1 = smf.logit("correct ~ confusable + log_load + C(model)", fit).fit(disp=0)
r2 = smf.logit("correct ~ target_similarity + saturation + C(model)", fit).fit(disp=0)
r0 = smf.logit("correct ~ C(model)", fit).fit(disp=0)
M["AfiveCoefConf"] = f"{r1.params['confusable']:.2f}"; M["AfivePConf"] = "< 0.001" if r1.pvalues["confusable"] < 1e-3 else f"{r1.pvalues['confusable']:.3f}"
M["AfiveCoefLoad"] = f"{r1.params['log_load']:.2f}"; M["AfivePLoad"] = f"{r1.pvalues['log_load']:.2f}"
M["AfiveCoefSim"] = f"{r2.params['target_similarity']:.2f}"; M["AfiveCoefSat"] = f"{r2.params['saturation']:.3f}"; M["AfivePSat"] = f"{r2.pvalues['saturation']:.3f}"
auc_sim = roc_auc_score(1 - test.correct, -r2.predict(test)); auc_conf = roc_auc_score(1 - test.correct, -r1.predict(test))
auc_base = roc_auc_score(1 - test.correct, -r0.predict(test))
M["AfiveAucSim"] = f"{auc_sim:.2f}"; M["AfiveAucConf"] = f"{auc_conf:.2f}"; M["AfiveAucBase"] = f"{auc_base:.2f}"
t = d.pivot_table(index="k", columns=["model", "confusable"], values="correct")
TAB.append("\\begin{table}[t]\\centering\\caption{Series A5 on real tool definitions (Toucan): share of correct choices (\\%) by toolkit size, "
           "model, and kind of distractors (random or confusable).}\\label{tab:a5real}\\scriptsize\\begin{tabular}{lrrrr}\\toprule "
           "$k$ & Haiku, random & Haiku, confusable & Opus, random & Opus, confusable \\\\ \\midrule")
for k, row in t.iterrows():
    TAB.append(f"{k} & " + " & ".join(f"{100 * row[(m, c)]:.1f}" for m in ["claude-haiku-5-5", "claude-opus-5-5"] for c in [0, 1]) + " \\\\")
TAB.append("\\bottomrule\\end{tabular}\\end{table}")

# audit review by two models
mine = {(r["server"], r["tool"]): int(r["out"]) for r in csv.DictReader(open(os.path.join(HERE, "data", "audit", "audit_check_claude.csv")))}
def val(v):
    s = str(v).strip().lower(); return 1 if s in ("1", "true") else 0 if s in ("0", "false") else None
lab = {}
for m, tag in [("claude-haiku-5-5", "Haiku"), ("claude-sonnet-5-5", "Sonnet")]:
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", "audit", f"labels_llm_{m}.jsonl"))]
    lab[m] = {(r["server"], r["tool"]): val((r["labels"] or {}).get("out")) for r in rows}
    dec = [k for k, v in lab[m].items() if v is not None]; a = [(lab[m][k], v) for k, v in mine.items() if lab[m].get(k) is not None]
    M[f"Au{tag}Dec"] = str(len(dec)); M[f"Au{tag}Agree"] = str(sum(x == y for x, y in a)); M[f"Au{tag}N"] = str(len(a))
both = [k for k in lab["claude-haiku-5-5"] if lab["claude-haiku-5-5"][k] is not None and lab["claude-sonnet-5-5"].get(k) is not None]
M["AuPair"] = str(sum(lab["claude-haiku-5-5"][k] == lab["claude-sonnet-5-5"][k] for k in both)); M["AuPairN"] = str(len(both))

# C1 with language models
c = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(LLM, "C1_*seed*.csv"))]); M["CoSeeds"] = str(c.seed.nunique())
cp, lg = c[c.method == "Compositio Prompto"], c[c.method == "LLM-gen+EmerKit"]
M["CoN"] = str(len(cp)); mac("CoCpEq", cp.equipped.mean()); mac("CoCpRisk", cp.any_dfr.mean()); mac("CoCpObj", cp.obj_cov.mean())
mac("CoLgValid", lg.parsed.mean()); v = lg[lg.parsed == 1]
mac("CoLgEq", v.equipped.mean()); mac("CoLgRisk", v.any_dfr.mean()); mac("CoLgObj", v.obj_cov.mean()); M["CoLgPrereq"] = str(int(v.prereq_viol.sum()))
mac("CoLgRaw", v.raw_valid.mean())

# F1 with an LLM agent
f = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(HERE, "results", "f1", "F1_llm*_seed*.csv"))]); M["FlSeeds"] = str(f.seed.nunique())
M["FlN"] = str(len(f)); mac("FlExposed", f.exposed_only_K.mean()); mac("FlBlocked", f.call_outside_blocked.mean())
M["FlUnexp"] = str(int(f.unexpected.sum())); mac("FlRealised", (f.observed.sum() / max(1, f.predicted.sum()))); M["FlDenied"] = str(int((f.denied_observed > 0).sum()))

# A6 on real tools
a6 = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(LLM, "A6real_*.csv"))])
g6 = a6.groupby("method").agg(success=("success", "mean"), avail=("available", "mean"), clones=("clone_pairs", "mean"))
for m, tag in [("EmerKit", "Emk"), ("RAG-MCP", "Rag"), ("Random", "Rnd")]:
    mac(f"Six{tag}", g6.loc[m, "success"]); mac(f"Six{tag}Avail", g6.loc[m, "avail"]); M[f"Six{tag}Clones"] = f"{g6.loc[m, 'clones']:.2f}"
pv = a6.pivot_table(index="probe", columns="method", values="success")
from scipy.stats import wilcoxon
M["SixP"] = f"{wilcoxon(pv['EmerKit'], pv['RAG-MCP']).pvalue:.2f}"; M["SixN"] = str(len(pv))

# panel sensitivity
p1 = add(L("P1")); t1 = p1.groupby(["profile", "method"]).agg(eq=("eq", "mean"), dfr=("dfr_e", "mean"), valid=("valid", "mean"))
for prof in ["standard", "panel"]:
    mac(f"Pone{prof.title()}Eq", t1.loc[(prof, "EmerKit"), "eq"]); mac(f"Pone{prof.title()}Dfr", t1.loc[(prof, "EmerKit"), "dfr"])
    mac(f"Pone{prof.title()}ZfDfr", t1.loc[(prof, "Zeng+filter"), "dfr"])
p2 = L("P2"); g2 = p2[p2.method == "EmerKit"].groupby("profile").agg(all_eq=("all_equipped", "mean"), risk=("any_dfr", "mean"), sub=("substituted", "mean"))
for prof in ["standard", "panel"]:
    mac(f"Ptwo{prof.title()}All", g2.loc[prof, "all_eq"]); mac(f"Ptwo{prof.title()}Risk", g2.loc[prof, "risk"]); mac(f"Ptwo{prof.title()}Sub", g2.loc[prof, "sub"])

with open(os.path.join(REP, "macros3.tex"), "w") as fo:
    fo.write("% generated by make_report3.py\n" + "".join(f"\\newcommand{{\\R{k}}}{{{v}}}\n" for k, v in sorted(M.items())))
with open(os.path.join(REP, "tables3.tex"), "w") as fo:
    fo.write("% generated by make_report3.py\n" + "\n".join(TAB) + "\n")
print(json.dumps(M, indent=0))
