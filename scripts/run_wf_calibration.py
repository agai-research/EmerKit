"""Workflow-level calibration (extension of Sect. 4.6). The loss of a calibration request is 1 when the delivered workflow
(after substitution) contains at least one toolkit with a denied flow under the true labels; Learn-then-Test with fixed-sequence
testing then certifies a level for whole workflows. Compared on held-out requests with the task-level level of series B1.
Usage: python scripts/run_wf_calibration.py SEED [N_CAL] [N_TEST]"""
import os, sys, json, time
import numpy as np, pandas as pd
from scipy.stats import binom
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT))
from emerkit.data import synth_catalogue, generate_requests, LABELS
from emerkit.prep import Prepared
from emerkit.vocab import norm_base
from emerkit import compose as C
from emerkit.e2e import run_workflow, generate_candidates, sub_catalogue
from run_pilot import GRID

ZETA, XIS = 0.10, (0.05, 0.10)


def memo_compose(P, labels_by_eps):
    """Caches compositions on (task, resolved labels of its candidates): levels often leave a task's labels unchanged."""
    cache, orig = {}, C.compose_task
    def f(prep, task, labels, deny, prm=None, exclude=(), explain=True):
        cands = [i for i in C.candidates_all(prep, task) if i not in exclude]
        key = (task["type"], task["domain"], tuple(sorted(task["R"])), tuple(sorted(task["D"])), tuple(sorted(task.get("delta_in", ()))),
               tuple(sorted(task.get("ho_in", ()))), tuple(sorted(task.get("ho_out", ()))), tuple(sorted(exclude)),
               tuple(tuple(sorted(labels[i].items())) for i in cands))
        if key not in cache:
            cache[key] = orig(prep, task, labels, deny, prm, exclude, explain=False)
        return cache[key]
    return f, cache


def deliver(P, wfs, lab, deny, truth, seed):
    out = []
    for k, wf in enumerate(wfs):
        wf = json.loads(json.dumps(wf, default=list))
        for t in wf["tasks"]:
            t["R"], t["D"] = set(t["R"]), set(t["D"])
        wf["edges"] = [(a, b, set(h)) for a, b, h in wf["edges"]]
        b = run_workflow(P, wf, lab, deny, C.DEFAULTS, np.random.default_rng(seed * 7919 + k), s_max=2, sub_cat=sub_catalogue("medium"))
        eq = [p for p in b["plan"].values() if p["K"]]
        risky = any(C.risk(P.tools, p["K"], truth, p["task"], deny) for p in eq)
        out.append(dict(risky=int(risky), all_equipped=int(not b["gaps"]), equipped=len(eq) / len(wf["tasks"]), substituted=len(b["subs"]) / len(wf["tasks"])))
    return out


def main(seed, n_cal=300, n_test=150):
    t0 = time.time()
    T, S = synth_catalogue(seed); P = Prepared(T, S, seed=seed); deny = norm_base(); truth = P.labels_for(None, "truth")
    rng = np.random.default_rng(seed + 3)
    reqs = generate_requests(seed + 9000, n_per_domain=(n_cal + n_test) // 3 + 1)
    wfs = [w for q in reqs for w in generate_candidates(rng, q, 1)][: n_cal + n_test]
    cal, test = wfs[:n_cal], wfs[n_cal:]
    lab_eps = {e: P.labels_for(P.conformal_sets(e), "worst") for e in GRID}
    C.compose_task, cache = memo_compose(P, lab_eps)
    import emerkit.e2e as E; E.compose_task = C.compose_task
    loss = {}
    eps_hat = {xi: 0.0 for xi in XIS}; open_xi = set(XIS)
    for e in GRID:                                            # fixed sequence, most conservative level first
        if not open_xi: break
        loss[e] = [r["risky"] for r in deliver(P, cal, lab_eps[e], deny, truth, seed)]
        for xi in list(open_xi):
            if binom.cdf(sum(loss[e]), n_cal, xi) <= ZETA:
                eps_hat[xi] = e
            else:
                open_xi.discard(xi)
    task_level = json.load(open(os.path.join(ROOT, "results", f"ehat_seed{seed}.json")))
    rows = []
    for name, e in [("workflow-level xi=0.05", eps_hat[0.05]), ("workflow-level xi=0.10", eps_hat[0.10]),
                    ("task-level xi=0.05", task_level["e_hat"]), ("task-level xi=0.10", task_level["e_hat10"])]:
        lab = lab_eps.get(e) or P.labels_for(P.conformal_sets(e), "worst")
        res = deliver(P, test, lab, deny, truth, seed)
        rows.append(dict(seed=seed, calibration=name, eps=e, workflow_risk=np.mean([r["risky"] for r in res]),
                         all_equipped=np.mean([r["all_equipped"] for r in res]), task_equipped=np.mean([r["equipped"] for r in res]),
                         substituted=np.mean([r["substituted"] for r in res]), n_test=len(res), n_cal=n_cal))
    d = pd.DataFrame(rows); out = os.path.join(ROOT, "results", "wf_calibration"); os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, f"wfcal_seed{seed}.csv"), index=False)
    print(f"seed {seed} in {time.time() - t0:.0f}s, compositions {len(cache)}"); print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]) if len(a) > 1 else 300, int(a[2]) if len(a) > 2 else 150)
