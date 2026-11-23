"""Aggregate pilot results: means with 95% bootstrap CIs (tasks pooled over seeds) and paired Wilcoxon tests vs EmerKit."""
import glob, os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
R = os.path.join(os.path.dirname(__file__), "results")
rng = np.random.default_rng(0)

def boot(x, B=10000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    idx = rng.integers(0, len(x), (B, len(x))); m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

def load(name):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(R, f"{name}_seed*.csv")))], ignore_index=True)

def a1_table():
    d = load("a1")
    d["valid"] = ((d["eq"] == 1) & (d["cov"] == 1) & (d["dfr"] == 0) & (d["qviol"] == 0)).astype(float)
    d["dfr_eq"] = np.where(d["eq"] == 1, d.dfr, np.nan)
    d["qviol_eq"] = np.where(d["eq"] == 1, d.qviol, np.nan)
    d["lb_eq"] = np.where(d["eq"] == 1, d.lb, np.nan); d["red_eq"] = np.where(d["eq"] == 1, d.red, np.nan)
    rows = []
    for m, g in d.groupby("method", sort=False):
        r = {"method": m}
        for k in ["eq", "cov", "valid", "dfr_eq", "qviol_eq", "red_eq", "lb_eq", "size"]:
            mu, lo, hi = boot(g[k]); r[k] = mu; r[k + "_lo"] = lo; r[k + "_hi"] = hi
        rows.append(r)
    tab = pd.DataFrame(rows)
    em = d[d.method == "EmerKit"].set_index(["seed", "task"]).sort_index()
    tests = []
    for m in d.method.unique():
        if m == "EmerKit": continue
        o = d[d.method == m].set_index(["seed", "task"]).sort_index()
        for k in ["valid", "cov"]:
            diff = em[k].values - o[k].values
            p = wilcoxon(em[k].values, o[k].values).pvalue if np.any(diff != 0) else 1.0
            tests.append(dict(method=m, metric=k, mean_diff=diff.mean(), p=p))
    return d, tab, pd.DataFrame(tests)

if __name__ == "__main__":
    d, tab, tests = a1_table()
    pd.set_option("display.width", 200)
    print(tab[["method", "eq", "cov", "valid", "valid_lo", "valid_hi", "dfr_eq", "qviol_eq", "red_eq", "lb_eq", "size"]].round(3).to_string(index=False))
    print(tests.round(4).to_string(index=False))
    b = load("b1"); print(b.groupby(["xi", "n_cal"])[["eps_hat", "certified", "dfr", "eq"]].mean().round(3).to_string())
    t = load("b1_trace"); print(t.groupby("eps")[["cal_viol_200", "test_eq", "test_dfr", "ambiguous"]].mean().round(3).T.to_string())
    x = load("d1"); x["dfr_eq"] = np.where(x["eq"] == 1, x.dfr, np.nan)
    x["valid"] = ((x["eq"] == 1) & (x["cov"] == 1) & (x["dfr"] == 0) & (x["qviol"] == 0)).astype(float)
    print(x.groupby("variant")[["eq", "valid", "dfr_eq", "qviol", "red", "lb"]].mean().round(3).to_string())
    a = load("a0b0"); print(a.groupby("metric").value.agg(["mean", "std"]).round(3).T.to_string())
