"""Statistics used in every series: bootstrap CIs, paired Wilcoxon tests with Holm correction, Cliff's delta,
and the mixed-effects logistic model of the selection probes (random intercept per model)."""
import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(x, B=10000, alpha=0.05, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(x), (B, len(x))); m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 100 * alpha / 2)), float(np.percentile(m, 100 * (1 - alpha / 2)))


def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum(); lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (len(a) * len(b)))


def holm(pvalues):
    p = np.asarray(pvalues, float); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def paired_tests(df, key, metric, reference, others):
    """Paired Wilcoxon signed-rank tests reference vs each other method on `metric`, matched on `key`; Holm-adjusted."""
    ref = df[df.method == reference].set_index(key)[metric]
    rows = []
    for m in others:
        o = df[df.method == m].set_index(key)[metric]
        j = ref.index.intersection(o.index); a, b = ref.loc[j].values, o.loc[j].values
        ok = ~(np.isnan(a) | np.isnan(b)); a, b = a[ok], b[ok]
        p = wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        rows.append(dict(method=m, n=len(a), mean_diff=float(np.mean(a - b)), cliffs_delta=cliffs_delta(a, b), p=float(p)))
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = float(q)
    return rows


def fit_probe_model(df):
    """Selection probes: P(correct) = logistic(b0 + b1*interference + b2*saturation + model random intercept).
    Returns fixed effects; lam_I = -b1/b0-scale and lam_C = -b2 are read in log-odds units."""
    import statsmodels.formula.api as smf
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    try:
        md = BinomialBayesMixedGLM.from_formula("correct ~ interference + saturation", {"model": "0 + C(model)"}, df)
        r = md.fit_vb()
        return dict(method="mixed (VB)", params=dict(zip(r.model.exog_names, r.fe_mean)), sd=dict(zip(r.model.exog_names, r.fe_sd)))
    except Exception as e:  # fallback: fixed model effects
        r = smf.logit("correct ~ interference + saturation + C(model)", df).fit(disp=0)
        return dict(method=f"logit with model fixed effects ({type(e).__name__})", params=r.params.to_dict(), sd=r.bse.to_dict())
