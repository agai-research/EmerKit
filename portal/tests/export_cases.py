"""Exports composition cases solved by the Python implementation, for the cross-check of the browser engine."""
import os, sys, json
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, ROOT)
from emerkit.data import synth_catalogue, sample_tasks, generate_requests
from emerkit.prep import Prepared
from emerkit.vocab import norm_base
from emerkit.compose import compose_task, DEFAULTS
from emerkit.e2e import generate_candidates, _task_view
T, S = synth_catalogue(0); P = Prepared(T, S, seed=0); lab = P.labels_for(P.conformal_sets(0.05), "worst")
cases = []
tasks = sample_tasks(0, 150)
for q, rq in enumerate(generate_requests(0, n_per_domain=6)):
    for wf in generate_candidates(np.random.default_rng(q), rq, 1):
        tasks += [_task_view(wf, i, "isolated", {}, lab, T) for i in range(len(wf["tasks"]))]
for prof in ["standard", "strict"]:
    deny = norm_base(prof)
    for k, t in enumerate(tasks):
        prm = dict(DEFAULTS) if k % 3 else {**DEFAULTS, "B": 1000, "k_max": 3}
        r = compose_task(P, t, lab, deny, prm, explain=True)
        cases.append(dict(profile=prof, params={kk: prm[kk] for kk in ["B", "k_max"]},
                          task=dict(type=t["type"], domain=t["domain"], concepts=list(t["concepts"]), R=sorted(t["R"]), D=sorted(t["D"]),
                                    delta_in=sorted(t.get("delta_in", set())), ho_out=sorted(t.get("ho_out", set())), ho_in=sorted(t.get("ho_in", set()))),
                          status=r["status"], obj=r["obj"], K=sorted(r["K"]) if r["K"] else None,
                          conflict=[("|".join(g) if len(g) > 1 else g[0]) for g in r["info"].get("conflict", [])] if r["status"] == "infeasible" else None))
json.dump(cases, open(os.path.join(ROOT, "portal", "tests", "cases.json"), "w"))
print(len(cases), "cases;", sum(c["obj"] is not None for c in cases), "feasible;", sum(c["status"] == "infeasible" for c in cases), "infeasible")
