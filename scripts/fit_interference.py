"""Links the selection probes to the composer: fits the interference index of Sect. 4.4,
P(correct) = logistic(a_model - lambda_I * sum_j sigmoid(beta * (sim(target, j) - omega0)) - lambda_C * S(load)),
by maximum likelihood on the fit half of the real-definition probes, and evaluates it on the test half (AUC).
Writes config/fitted_weights.json. Usage: python scripts/fit_interference.py"""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_probes_real as RP
from emerkit.prep import Prepared

T, rows = RP.design()
S = RP.toucan_catalogue(0)[1]; P = Prepared(T, S, seed=0)
sims = {r["probe"]: np.array([P.desc_sim(r["target"], j) for j in r["toolkit"] if j != r["target"]]) for r in rows}
sat = {r["probe"]: r["saturation"] for r in rows}
d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "results", "llm", "A5real_*.csv"))])
models = sorted(d.model.unique())

def predict(theta, df):
    a = theta[:len(models)]; lam, beta, om, lamc = theta[len(models):]
    out = []
    for _, r in df.iterrows():
        inter = np.sum(1 / (1 + np.exp(-beta * (sims[r.probe] - om))))
        z = a[models.index(r.model)] - lam * inter - lamc * sat[r.probe]
        out.append(1 / (1 + np.exp(-z)))
    return np.clip(np.array(out), 1e-6, 1 - 1e-6)

def nll(theta, df):
    p = predict(theta, df); y = df.correct.values
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

fit, test = d[d.split == "fit"], d[d.split == "test"]
x0 = np.array([3.0] * len(models) + [1.0, 10.0, 0.5, 0.0])
bounds = [(-10, 15)] * len(models) + [(0, 50), (0.5, 80), (0.0, 1.0), (-1, 1)]
best = min((minimize(nll, x0 * s, args=(fit,), method="L-BFGS-B", bounds=bounds) for s in (0.5, 1.0, 1.5)), key=lambda r: r.fun)
th = best.x; lam, beta, om, lamc = th[len(models):]
auc = roc_auc_score(1 - test.correct, 1 - predict(th, test))
base = minimize(lambda a: nll(np.r_[a, 0.0, 10.0, 0.5, 0.0], fit), [3.0] * len(models), method="L-BFGS-B").x
auc0 = roc_auc_score(1 - test.correct, 1 - predict(np.r_[base, 0.0, 10.0, 0.5, 0.0], test))
res = dict(source="selection probes on real Toucan definitions (Claude models only)", models=models, lam_I=float(lam), beta=float(beta),
           omega0=float(om), lam_C=float(lamc), intercepts=dict(zip(models, map(float, th[:len(models)]))),
           test_auc=float(auc), test_auc_without_interference=float(auc0), n_fit=len(fit), n_test=len(test),
           note="lam_I and lam_C are in log-odds units; the composer's results are insensitive to lam_I in [0, 2] (series D2)")
json.dump(res, open(os.path.join(ROOT, "config", "fitted_weights.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
