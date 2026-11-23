"""Ground-truth evaluation of a toolkit for a task (external measures only; Q itself is never a comparison metric)."""
import math
from .compose import risk, task_bounds, candidates_all


def evaluate(prep, task, K, W, deny, prm, truth_labels):
    T, S = prep.tools, prep.servers
    R = task["R"]
    if K is None:
        return dict(eq=0, cov=0.0, size=0, load=0, lb=0.0, over=0, cp=0, red=0, qviol=None, dfr=None, real=0)
    covered = {r for r in R if any(T[i]["true_caps"] is not None and r in T[i]["true_caps"] for i in K)}
    servers = {T[i]["server"] for i in K}
    load = sum(T[i]["cost"] for i in K) + sum(S[s]["preamble"] for s in servers)
    dup = [T[i]["dup"] for i in K if T[i]["dup"] is not None]
    cp = sum(1 for k in set(dup) if dup.count(k) >= 2)
    red = sum(max(0, sum(1 for i in K if T[i]["true_caps"] and r in T[i]["true_caps"]) - 1) for r in R)
    # QoS of the serving assignment (method's own if given, else fastest true provider per capability)
    if W is None:
        W = {}
        for r in R:
            prov = [i for i in K if T[i]["true_caps"] and r in T[i]["true_caps"]] or [i for i in K if r in T[i]["caps"]]
            if prov:
                W[r] = min(prov, key=lambda i: T[i]["lat"])
    qviol = None
    if len(W) == len(R):
        latb, prb, depb = task_bounds(prep, task, candidates_all(prep, task), prm["tau"])
        lat = sum(T[i]["lat"] for i in W.values()); pr = sum(T[i]["price"] for i in W.values())
        dep = math.prod(S[s]["avail"] for s in {T[i]["server"] for i in W.values()}) * math.prod(T[i]["rel"] for i in W.values())
        qviol = int(lat > latb * 1.000001 or pr > prb * 1.000001 or dep < depb * 0.999999)
    rk = risk(T, K, truth_labels, task, deny)
    return dict(eq=1, cov=len(covered) / len(R) if R else 1.0, size=len(K), load=load, lb=load / prm["B"],
                over=int(load > prm["theta"] * prm["B"]), cp=cp, red=red, qviol=qviol,
                dfr=None if rk is None else int(rk > 0), real=int(any(not T[i]["synthetic"] for i in K)))
