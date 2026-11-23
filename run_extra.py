"""Additional series. X1: catalogue size; X2: capabilities required per task; X4: workflow length; X5: exact vs greedy composition;
X6: calibration under distribution shift. Usage: python run_extra.py SERIES CONFIG [...] ; python run_extra.py list"""
import sys, os, time, json
import numpy as np, pandas as pd
from scipy.stats import binom
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from emerkit.data import synth_catalogue, sample_tasks
from emerkit.prep import Prepared
from emerkit.vocab import norm_base, TYPE_CAPS, CAPS, DOMAIN_CONCEPTS, DOMAINS
from emerkit.compose import compose_task, greedy_task, rag_mcp, zeng, DEFAULTS, risk
from emerkit.metrics import evaluate
from emerkit.e2e import run_workflow, sub_catalogue, _task_view
OUT = os.path.join(HERE, "results", "extra"); os.makedirs(OUT, exist_ok=True)
SEEDS, EPS = [0, 1, 2], 0.05
GRIDS = {
 "X1": [dict(providers=p) for p in [(1, 2), (2, 3), (4, 6), (8, 10), (12, 16), (18, 22)]],
 "X2": [dict(nR=n) for n in [1, 2, 3, 4, 5]],
 "X4": [dict(L=l) for l in [3, 4, 5, 6, 8, 10]],
 "X5": [dict(N=n, nR=3) for n in [10, 20, 50, 100]] + [dict(N=20, nR=n) for n in [1, 2, 4, 5]],
 "X6": [dict(scenario=s) for s in ["matched", "types: low-stakes to assessment", "types: assessment to low-stakes",
                                   "domains: programming+language to data science", "domains: data science to programming+language"]],
}

def all_methods(P, task, lab, decl, truth, deny, prm):
    out = []
    for name, labels in [("EmerKit", lab), ("Oracle", truth)]:
        r = compose_task(P, task, labels, deny, prm, explain=False); out.append((name, r["K"], r["w"], r["time"]))
    for name, fn in [("RAG-MCP", lambda: rag_mcp(P, task, prm["k_max"])), ("RAG-MCP+filter", lambda: rag_mcp(P, task, prm["k_max"], decl, deny)),
                     ("Zeng", lambda: zeng(P, task, prm)), ("Zeng+filter", lambda: zeng(P, task, prm, decl, deny))]:
        t = time.time(); K = fn(); out.append((name, K, None, time.time() - t))
    return out

def with_R(task, n, rng):
    pool = [c for c in TYPE_CAPS[task["type"]] if CAPS[c][1] is None or task["domain"] in CAPS[c][1]]
    return dict(task, R=set(rng.choice(pool, min(n, len(pool)), replace=False)))

def x_tasklevel(series, ci, cfg):
    rows = []; prm = dict(DEFAULTS); deny = norm_base()
    for seed in SEEDS:
        T, S = synth_catalogue(seed, providers=cfg["providers"]) if series == "X1" else synth_catalogue(seed)
        P = Prepared(T, S, seed=seed); rng = np.random.default_rng(seed)
        lab = P.labels_for(P.conformal_sets(EPS), "worst"); truth = P.labels_for(None, "truth"); decl = P.labels_for(None, "declared")
        for k, task in enumerate(sample_tasks(seed, 120)):
            if series == "X2":
                task = with_R(task, cfg["nR"], rng); prm = {**DEFAULTS, "k_max": max(DEFAULTS["k_max"], cfg["nR"])}   # same size allowance for all methods
            for name, K, W, dt in all_methods(P, task, lab, decl, truth, deny, prm):
                rows.append(dict(series=series, config=ci, n_tools=len(T), nR=len(task["R"]), seed=seed, task=k, type=task["type"], method=name, time=dt,
                                 **evaluate(P, task, K, W, deny, prm, truth)))
    return rows

def x_workflow(series, ci, cfg):
    rows = []; prm = dict(DEFAULTS); deny = norm_base(); L = cfg["L"]
    for seed in SEEDS:
        T, S = synth_catalogue(seed); P = Prepared(T, S, seed=seed)
        lab = P.labels_for(P.conformal_sets(EPS), "worst"); truth = P.labels_for(None, "truth"); decl = P.labels_for(None, "declared")
        pool = sample_tasks(seed + 7000, 600); rng = np.random.default_rng(seed + 11)
        rank = {d: {c: i for i, (c, _) in enumerate(DOMAIN_CONCEPTS[d])} for d in DOMAINS}
        for q in range(40):
            dom = DOMAINS[q % len(DOMAINS)]; cand = [t for t in pool if t["domain"] == dom]
            tasks = [dict(cand[i]) for i in rng.choice(len(cand), L, replace=False)]
            tasks.sort(key=lambda t: min((rank[dom].get(c, 0) for c in t["concepts"]), default=0))
            wf = dict(domain=dom, targets=tasks[-1]["concepts"] or [DOMAIN_CONCEPTS[dom][0][0]], tasks=tasks, edges=[(i, i + 1, set()) for i in range(L - 1)])
            for name, smax in [("EmerKit", 2), ("EmerKit-nosub", 0)]:
                import copy
                b = run_workflow(P, copy.deepcopy(wf), lab, deny, prm, np.random.default_rng(seed * 100 + q), s_max=smax, sub_cat=sub_catalogue("medium"))
                eq = [p for p in b["plan"].values() if p["K"]]
                rows.append(dict(series=series, config=ci, L=L, seed=seed, wf=q, method=name, equipped=len(eq) / L, all_equipped=int(not b["gaps"]),
                                 any_dfr=int(any(risk(T, p["K"], truth, p["task"], deny) for p in eq)), substituted=len(b["subs"]) / L))
            for name in ["RAG-MCP", "RAG-MCP+filter", "Zeng", "Zeng+filter"]:
                ok, d = 0, []
                for i in range(L):
                    v = _task_view(wf, i, "isolated", {}, lab, T)
                    K = (rag_mcp(P, v, 4, decl if "filter" in name else None, deny) if "RAG" in name else zeng(P, v, prm, decl if "filter" in name else None, deny))
                    if K:
                        e = evaluate(P, v, K, None, deny, prm, truth); good = e["cov"] == 1 and e["qviol"] == 0
                        ok += good; d.append(e["dfr"])
                rows.append(dict(series=series, config=ci, L=L, seed=seed, wf=q, method=name, equipped=ok / L, all_equipped=int(ok == L), any_dfr=int(any(d)), substituted=0.0))
    return rows

def x_greedy(series, ci, cfg):
    rows = []; prm = {**DEFAULTS, "N": cfg["N"]}; deny = norm_base()
    for seed in SEEDS:
        T, S = synth_catalogue(seed, providers=(12, 16)); P = Prepared(T, S, seed=seed); rng = np.random.default_rng(seed)
        lab = P.labels_for(P.conformal_sets(EPS), "worst")
        for k, task in enumerate(sample_tasks(seed, 90)):
            task = with_R(task, cfg["nR"], rng)
            e = compose_task(P, task, lab, deny, prm, explain=False); g = greedy_task(P, task, lab, deny, prm)
            rows.append(dict(series=series, config=ci, **cfg, seed=seed, task=k, exact_ok=int(e["K"] is not None), greedy_ok=int(g["K"] is not None),
                             exact_obj=e["obj"], greedy_obj=g["obj"], exact_s=e["time"], greedy_s=g["time"]))
    return rows

def x_shift(series, ci, cfg):
    from run_pilot import GRID
    rows = []; prm = dict(DEFAULTS); deny = norm_base(); sc = cfg["scenario"]
    low, high = {"acquisition", "practice", "discussion"}, {"formative", "summative", "production"}
    sel = {"matched": (lambda t: True, lambda t: True),
           "types: low-stakes to assessment": (lambda t: t["type"] in low, lambda t: t["type"] in high),
           "types: assessment to low-stakes": (lambda t: t["type"] in high, lambda t: t["type"] in low),
           "domains: programming+language to data science": (lambda t: t["domain"] != "datascience", lambda t: t["domain"] == "datascience"),
           "domains: data science to programming+language": (lambda t: t["domain"] == "datascience", lambda t: t["domain"] != "datascience")}[sc]
    for seed in SEEDS:
        T, S = synth_catalogue(seed); P = Prepared(T, S, seed=seed); truth = P.labels_for(None, "truth")
        pc, pt = sample_tasks(seed + 5000, 600), sample_tasks(seed, 600)          # pools are ordered by type: shuffle before slicing
        r_ = np.random.default_rng(seed + 61); r_.shuffle(pc); r_.shuffle(pt)
        cal = [t for t in pc if sel[0](t)][:300]; test = [t for t in pt if sel[1](t)][:300]
        cache = {}
        def outcome(task, k, tag, eps, lab):
            cands = [i for i, t in enumerate(T) if t["caps"] & task["R"]]
            key = (tag, k, tuple(tuple(sorted(lab[i].items())) for i in cands))
            if key not in cache:
                r = compose_task(P, task, lab, deny, prm, explain=False)
                cache[key] = (r["K"] is not None, int(bool(r["K"]) and risk(T, r["K"], truth, task, deny) > 0))
            return cache[key]
        ehat = {0.05: 0.0, 0.10: 0.0}; open_xi = {0.05, 0.10}; res = {}
        for eps in GRID:
            lab = P.labels_for(P.conformal_sets(eps), "worst")
            res[eps] = lab
            viol = sum(outcome(t, k, "cal", eps, lab)[1] for k, t in enumerate(cal))
            for xi in list(open_xi):
                if binom.cdf(viol, len(cal), xi) <= 0.10: ehat[xi] = eps
                else: open_xi.discard(xi)
            if not open_xi: break
        for xi in (0.05, 0.10):
            lab = res.get(ehat[xi]) or P.labels_for(P.conformal_sets(ehat[xi]), "worst")
            o = [outcome(t, k, "test", ehat[xi], lab) for k, t in enumerate(test)]
            eq = sum(a for a, _ in o); bad = sum(b for _, b in o)
            rows.append(dict(series=series, config=ci, scenario=sc, seed=seed, xi=xi, eps_hat=ehat[xi], n_cal=len(cal), n_test=len(test),
                             equipped=eq / len(test), dfr=bad / max(1, eq)))
    return rows

RUN = {"X1": x_tasklevel, "X2": x_tasklevel, "X4": x_workflow, "X5": x_greedy, "X6": x_shift}
if __name__ == "__main__":
    if sys.argv[1] == "list":
        print({k: len(v) for k, v in GRIDS.items()}); sys.exit()
    s = sys.argv[1]
    for c in sys.argv[2:]:
        t0 = time.time(); cfg = GRIDS[s][int(c)]; rows = RUN[s](s, int(c), cfg)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{s}_c{c}.csv"), index=False); print(f"{s} c{c} {cfg} {time.time() - t0:.0f}s", flush=True)
