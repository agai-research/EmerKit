"""Agreement of the LLM expert panel with the provisional pedagogical parameters and between models (Fleiss' kappa)."""
import glob, os, numpy as np, pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "llm")
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "judges_*.csv"))])
def b(v):
    if isinstance(v, str): return {"true": 1.0, "false": 0.0, "True": 1.0, "False": 0.0}.get(v.strip(), np.nan)
    return float(v) if v == v else np.nan
out = {}
for kind in ["sub", "norm"]:
    x = d[d.kind == kind].copy(); x["v"] = x.value.map(b); x["c"] = x.current.map(b)
    p = x.pivot_table(index=["a", "b"], columns="model", values="v", aggfunc="first").dropna()
    cur = x.groupby(["a", "b"]).c.first().loc[p.index]
    maj = (p.mean(axis=1) > 0.5).astype(float)
    tab, _ = aggregate_raters(p.astype(int).values)
    out[kind] = dict(items=len(p), unanimous=float((p.nunique(axis=1) == 1).mean()), fleiss_kappa=float(fleiss_kappa(tab)),
                     majority_agrees_with_current=float((maj == cur).mean()),
                     disagreements=[(i, int(cur.loc[i]), int(maj.loc[i])) for i in p.index[maj != cur]])
f = d[d.kind == "fit"].copy(); f["v"] = pd.to_numeric(f.value, errors="coerce")
p = f.pivot_table(index=["a", "b"], columns="model", values="v").dropna(); cur = pd.to_numeric(f.groupby(["a", "b"]).current.first(), errors="coerce").loc[p.index]
out["fit"] = dict(items=len(p), corr_with_current=float(np.corrcoef(p.mean(axis=1), cur)[0, 1]), mean_abs_gap=float((p.mean(axis=1) - cur).abs().mean()),
                  inter_model_corr=float(np.mean([np.corrcoef(p[a], p[b_])[0, 1] for i, a in enumerate(p) for b_ in list(p)[i + 1:]])),
                  largest_gaps={str(k): round(v, 2) for k, v in (p.mean(axis=1) - cur).sort_values(key=abs, ascending=False).head(6).items()})
if __name__ == "__main__":
    import json; print(json.dumps(out, indent=1, default=str))
