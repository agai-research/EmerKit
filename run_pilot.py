"""Pilot runner (synthetic catalogue, no LLM): series A0, B0, A1, B1, D1. Usage: python run_pilot.py SEED [SEED ...]"""
import sys, json, time, os
import numpy as np, pandas as pd
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(__file__))
from emerkit.data import synth_catalogue, sample_tasks, LABELS
from emerkit.prep import Prepared
from emerkit.vocab import norm_base
from emerkit.compose import compose_task, rag_mcp, zeng, DEFAULTS, risk, candidates_all
from emerkit.metrics import evaluate

OUT = os.path.join(os.path.dirname(__file__), "results"); os.makedirs(OUT, exist_ok=True)
GRID = sorted(set([0.0, 0.001, 0.0025, 0.005, 0.01, 0.015, 0.02] + [round(0.025 * k, 3) for k in range(1, 21)]))   # finer near 0
XIS, NCALS, ZETA = [0.05, 0.10, 0.20], [45, 100, 200, 300], 0.10
N_CAL = 300                      # calibration tasks used by A1 and D1 (power: 9 tolerated violations at xi = 0.05)


def ltt(viol_by_eps, n, xi, zeta):
    eps_hat = 0.0
    for eps in GRID:
        if binom.cdf(viol_by_eps[eps], n, xi) <= zeta:
            eps_hat = eps
        else:
            break
    return eps_hat


def run_seed(seed, n_test=300, n_cal_max=300, part='all'):
    t0 = time.time()
    T, S = synth_catalogue(seed); P = Prepared(T, S, seed=seed)
    deny = norm_base("standard"); prm = dict(DEFAULTS)
    truth = P.labels_for(None, "truth"); decl = P.labels_for(None, "declared")
    rows_ab = []
    # ---------------- A0: capability extraction; B0: label inference and conformal coverage (tools outside the calibration set)
    test_idx = [i for i in range(len(T)) if i not in P.cal_set]
    tp = sum(len(T[i]["caps"] & T[i]["true_caps"]) for i in test_idx); fp = sum(len(T[i]["caps"] - T[i]["true_caps"]) for i in test_idx)
    fn = sum(len(T[i]["true_caps"] - T[i]["caps"]) for i in test_idx)
    rows_ab.append(dict(seed=seed, series="A0", metric="precision", value=tp / max(1, tp + fp)))
    rows_ab.append(dict(seed=seed, series="A0", metric="recall", value=tp / max(1, tp + fn)))
    for l in LABELS:
        y = np.array([T[i]["truth"][l] for i in test_idx]); p = P.p1[l][test_idx]
        rows_ab.append(dict(seed=seed, series="B0", metric=f"acc_{l}", value=float(np.mean((p >= 0.5) == y))))
    for eps in [0.05, 0.1, 0.2]:
        G = P.conformal_sets(eps)
        cov = np.mean([T[i]["truth"][l] in G[l][i] for l in LABELS for i in test_idx])
        amb = np.mean([G[l][i] == {0, 1} for l in LABELS for i in test_idx])
        rows_ab.append(dict(seed=seed, series="B0", metric=f"coverage_eps{eps}", value=float(cov)))
        rows_ab.append(dict(seed=seed, series="B0", metric=f"ambiguous_eps{eps}", value=float(amb)))
    # ---------------- B1: calibration and test tasks, compositions cached per level
    cal_tasks = sample_tasks(seed + 5000, n_cal_max)[:n_cal_max]
    rng = np.random.default_rng(seed); order = rng.permutation(len(cal_tasks)); cal_tasks = [cal_tasks[i] for i in order]
    test_tasks = sample_tasks(seed, n_test)
    ehat_file = os.path.join(OUT, f"ehat_seed{seed}.json")
    if part == "b":
        eh = json.load(open(ehat_file)); e_hat, e_hat10 = eh["e_hat"], eh["e_hat10"]
        lab_eps, test_res = {}, {}
        for e in {e_hat, e_hat10}:
            lab_eps[e] = P.labels_for(P.conformal_sets(e), "worst")
            test_res[e] = [compose_task(P, t, lab_eps[e], deny, prm, explain=False) for t in test_tasks]
        return _part_b(seed, P, T, deny, prm, truth, decl, test_tasks, test_res, lab_eps, e_hat, t0, e_hat10)
    lab_eps = {eps: P.labels_for(P.conformal_sets(eps), "worst") for eps in GRID}
    cal_loss = {eps: [] for eps in GRID}; test_res = {eps: [] for eps in GRID}
    cache = {}
    def cached(kind, k, task, eps):
        # a composition only depends on the resolved labels of the task's candidates: reuse it when they do not change
        cands = candidates_all(P, task)
        sig = (kind, k, tuple(tuple(sorted(lab_eps[eps][i].items())) for i in cands))
        if sig not in cache:
            cache[sig] = compose_task(P, task, lab_eps[eps], deny, prm, explain=False)
        return cache[sig]
    for eps in GRID:
        for k, task in enumerate(cal_tasks):
            r = cached("cal", k, task, eps)
            cal_loss[eps].append(int(bool(r["K"]) and bool(risk(T, r["K"], truth, task, deny))))
        for k, task in enumerate(test_tasks):
            test_res[eps].append(cached("test", k, task, eps))
    rows_b1 = []
    for xi in XIS:
        for n in NCALS:
            viol = {eps: sum(cal_loss[eps][:n]) for eps in GRID}
            e_hat = ltt(viol, n, xi, ZETA)
            ev = [evaluate(P, t, r["K"], r["w"], deny, prm, truth) for t, r in zip(test_tasks, test_res[e_hat])]
            rows_b1.append(dict(seed=seed, xi=xi, n_cal=n, eps_hat=e_hat, certified=int(binom.cdf(viol[e_hat], n, xi) <= ZETA),
                                dfr=np.mean([e["dfr"] for e in ev if e["eq"]]) if any(e["eq"] for e in ev) else np.nan,
                                dfr_all=np.mean([e["dfr"] if e["eq"] else 0 for e in ev]), eq=np.mean([e["eq"] for e in ev])))
    trace = []
    for eps in GRID:
        ev = [evaluate(P, t, r["K"], r["w"], deny, prm, truth) for t, r in zip(test_tasks, test_res[eps])]
        eqv = [e for e in ev if e["eq"]]
        trace.append(dict(seed=seed, eps=eps, cal_viol=sum(cal_loss[eps]), test_eq=np.mean([e["eq"] for e in ev]),
                          test_dfr=np.mean([e["dfr"] for e in eqv]) if eqv else np.nan,
                          ambiguous=float(np.mean([P.conformal_sets(eps)[l][i] == {0, 1} for l in LABELS for i in test_idx]))))
    e_hat = ltt({eps: sum(cal_loss[eps][:N_CAL]) for eps in GRID}, N_CAL, 0.05, ZETA)
    e_hat10 = ltt({eps: sum(cal_loss[eps][:N_CAL]) for eps in GRID}, N_CAL, 0.10, ZETA)
    json.dump({"e_hat": e_hat, "e_hat10": e_hat10}, open(ehat_file, "w"))
    for name, rows in [("a0b0", rows_ab), ("b1", rows_b1), ("b1_trace", trace)]:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{name}_seed{seed}.csv"), index=False)
    if part == "a":
        print(f"seed {seed} part a done in {time.time() - t0:.0f}s, eps_hat={e_hat}", flush=True); return
    return _part_b(seed, P, T, deny, prm, truth, decl, test_tasks, test_res, lab_eps, e_hat, t0, e_hat10)


def _part_b(seed, P, T, deny, prm, truth, decl, test_tasks, test_res, lab_eps, e_hat, t0, e_hat10=None):
    # ---------------- A1: reference comparison at the calibrated level (xi=0.05, n_cal=100)
    rows_a1 = []
    for k, task in enumerate(test_tasks):
        base = dict(seed=seed, task=k, type=task["type"], domain=task["domain"], nR=len(task["R"]), eps_hat=e_hat)
        r = test_res[e_hat][k]
        rows_a1.append({**base, "method": "EmerKit", "status": r["status"], "solve_s": r["time"], **evaluate(P, task, r["K"], r["w"], deny, prm, truth)})
        if e_hat10 is not None:
            r10 = test_res[e_hat10][k]
            rows_a1.append({**base, "eps_hat": e_hat10, "method": "EmerKit-xi0.10", "status": r10["status"], "solve_s": r10["time"],
                            **evaluate(P, task, r10["K"], r10["w"], deny, prm, truth)})
        ro = compose_task(P, task, truth, deny, prm, explain=False)        # oracle: true labels (for false rejections)
        rows_a1.append({**base, "method": "Oracle", "status": ro["status"], "solve_s": ro["time"], **evaluate(P, task, ro["K"], ro["w"], deny, prm, truth)})
        for name, fn in [("RAG-MCP", lambda: rag_mcp(P, task, prm["k_max"])),
                         ("RAG-MCP+filter", lambda: rag_mcp(P, task, prm["k_max"], decl, deny)),
                         ("Zeng", lambda: zeng(P, task, prm)),
                         ("Zeng+filter", lambda: zeng(P, task, prm, decl, deny))]:
            t1 = time.time(); K = fn()
            rows_a1.append({**base, "method": name, "status": "ok" if K is not None else "none", "solve_s": time.time() - t1,
                            **evaluate(P, task, K, None, deny, prm, truth)})
    # ---------------- D1: ablations at the same level
    lab = lab_eps[e_hat]
    variants = {"additive utility": dict(additive=True), "no interactions": dict(lam_I=0.0, lam_C=0.0),
                "no QoS": dict(use_qos=False), "declared annotations": "declared"}
    rows_d1 = []
    for vname, v in variants.items():
        for k, task in enumerate(test_tasks):
            if v == "declared":
                r = compose_task(P, task, decl, deny, prm, explain=False)
            else:
                r = compose_task(P, task, lab, deny, {**prm, **v}, explain=False)
            rows_d1.append(dict(seed=seed, task=k, type=task["type"], variant=vname, status=r["status"], solve_s=r["time"],
                                **evaluate(P, task, r["K"], r["w"], deny, prm, truth)))
    for name, rows in [("a1", rows_a1), ("d1", rows_d1)]:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{name}_seed{seed}.csv"), index=False)
    print(f"seed {seed} part b done in {time.time() - t0:.0f}s, eps_hat={e_hat}", flush=True)


if __name__ == "__main__":
    part = os.environ.get("PART", "all")
    for s in sys.argv[1:]:
        run_seed(int(s), part=part)
