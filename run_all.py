"""All LLM-free series. Usage: python run_all.py SERIES CONFIG_INDEX [...]; python run_all.py list"""
import sys, os, time, resource, copy
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from emerkit.data import synth_catalogue, real_catalogue, sample_tasks, generate_requests, archetypes, make_task, LABELS
from emerkit.prep import Prepared
from emerkit.sources import toucan_catalogue
from emerkit.vocab import norm_base, TYPE_CAPS, CAPS
from emerkit.compose import compose_task, rag_mcp, zeng, DEFAULTS, risk
from emerkit.metrics import evaluate
from emerkit.e2e import run_request, classify_archetypes, generate_candidates, _task_view, workflow_measures

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "grid"); os.makedirs(OUT, exist_ok=True)
REAL = os.path.join(HERE, "data", "mcp_tools_compact.json")
SEEDS, NT, EPS = [0, 1, 2], 150, 0.05

def one_factor(base, **axes):
    out = [dict(base)]
    for k, vals in axes.items():
        out += [{**base, k: v} for v in vals if v != base.get(k)]
    return out

GRIDS = {
 "A2": [dict(rho_conf=r) for r in [0.0, 0.1, 0.2, 0.3, 0.4]],
 "A3": [dict(B=b, length_factor=f) for f in [1, 4] for b in [500, 1000, 2000, 4000]],
 "A4": [dict(tau=t) for t in [0.5, 0.75, 1.0, 1.25, 1.5]],
 "B2": one_factor(dict(m=0.6, w=0.05, cue=0.4), m=[0.0, 0.2, 0.4, 0.8], w=[0.0, 0.1, 0.2], cue=[0.2, 0.6]),
 "B3": [dict(profile=p, ext_bias=e) for p in ["permissive", "standard", "strict"] for e in [0.5, 1.0, 2.0]],
 "D2": one_factor(dict(lam_I=1.0, lam_C=1.0, lam_Q=0.5, theta=0.75), lam_I=[0.0, 0.5, 2.0], lam_C=[0.0, 0.5, 2.0], lam_Q=[0.0, 1.0], theta=[0.5, 0.9]),
 "C3": one_factor(dict(N=20, nR=3, k_max=4, T_max=10.0), N=[10, 30, 50, 100], nR=[1, 2, 5], k_max=[3, 6, 8], T_max=[1.0, 60.0]),
 "C4": [dict(size=s) for s in [300, 1000, 2000, 2960]],
 "C2": one_factor(dict(s_max=2, density="medium", nu=0.5, mode="isolated"), s_max=[0, 1, 3], density=["low", "high"], nu=[0.1, 1.0], mode=["conservative"]),
 "E1": [dict(catalogue=c, profile=p) for c in ["synthetic", "mixed"] for p in ["permissive", "standard", "strict"]],
 "E2": [dict(remove=r, annotate=0.0) for r in [0.1, 0.2, 0.3]] + [dict(remove=0.0, annotate=a) for a in [0.25, 0.5]],
}
import itertools
def product(**axes):
    keys = list(axes); return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]
GRIDS_FULL = {      # full factorial designs of the protocol (EMERKIT_GRID=full); run in Claude Code on several cores
 "A2": product(rho_conf=[0.0, 0.1, 0.2, 0.3, 0.4], dup_paraphrase=[0.0, 0.3, 0.6]),
 "A3": product(B=[500, 1000, 2000, 4000, 8000], length_factor=[1, 2, 4, 8]),
 "A4": product(tau=[0.5, 0.75, 1.0, 1.25, 1.5]),
 "B2": product(m=[0.0, 0.2, 0.4, 0.6, 0.8], w=[0.0, 0.05, 0.1, 0.2], cue=[0.2, 0.4, 0.6]),
 "B3": product(profile=["permissive", "standard", "strict"], ext_bias=[0.5, 1.0, 2.0]),
 "D2": product(lam_I=[0.0, 0.5, 1.0, 2.0], lam_C=[0.0, 0.5, 1.0, 2.0], lam_Q=[0.0, 0.5, 1.0], theta=[0.5, 0.75, 0.9]),
 "C3": product(N=[10, 20, 30, 50, 100], nR=[1, 2, 3, 5], k_max=[3, 4, 6, 8], T_max=[1.0, 10.0, 60.0]),
 "C4": product(size=[300, 1000, 3000, 5000, 8800]),
 "C2": product(s_max=[0, 1, 2, 3], density=["low", "medium", "high"], nu=[0.1, 0.5, 1.0], mode=["isolated", "conservative"]),
 "E1": product(catalogue=["synthetic", "mcp-tools", "toucan"], profile=["permissive", "standard", "strict"]),
 "E2": [dict(remove=r, annotate=0.0) for r in [0.1, 0.2, 0.3]] + [dict(remove=0.0, annotate=a) for a in [0.25, 0.5]],
}
SENS = {"P1": [dict(profile="standard"), dict(profile="panel")],                       # sensitivity to the LLM panel's norms
        "P2": [dict(s_max=2, density="medium", nu=0.5, mode="isolated", profile="standard"),
               dict(s_max=2, density="panel", nu=0.5, mode="isolated", profile="panel")]}   # ... and substitution catalogue
GRIDS.update(SENS); GRIDS_FULL.update(SENS)
if os.environ.get("EMERKIT_GRID") == "full":
    GRIDS = GRIDS_FULL
    NT_FULL = True
METHODS = {"A2": "core", "A3": "core", "A4": "core", "B2": "safety", "B3": "safety", "D2": "emerkit", "P1": "safety"}
DATA_KEYS = ("rho_conf", "length_factor", "m", "w", "cue", "ext_bias", "dup_paraphrase")


def methods_for(kind, P, task, lab, decl, truth, deny, prm):
    out = []
    def em(name, labels, extra=None):
        r = compose_task(P, task, labels, deny, {**prm, **(extra or {})}, explain=False)
        out.append((name, r["K"], r["w"], r["time"]))
    em("EmerKit", lab)
    if kind == "core":
        em("EmerKit-no-interactions", lab, dict(lam_I=0.0, lam_C=0.0))
        t = time.time(); out.append(("RAG-MCP", rag_mcp(P, task, prm["k_max"]), None, time.time() - t))
        t = time.time(); out.append(("Zeng", zeng(P, task, prm), None, time.time() - t))
    if kind == "safety":
        em("Oracle", truth); em("EmerKit-declared", decl)
        t = time.time(); out.append(("RAG-MCP+filter", rag_mcp(P, task, prm["k_max"], decl, deny), None, time.time() - t))
        t = time.time(); out.append(("Zeng+filter", zeng(P, task, prm, decl, deny), None, time.time() - t))
    return out


def task_level(series, ci, cfg):
    rows = []; prm = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
    deny = norm_base(cfg.get("profile", "standard"))
    for seed in SEEDS:
        T, S = synth_catalogue(seed, **{k: v for k, v in cfg.items() if k in DATA_KEYS}); P = Prepared(T, S, seed=seed)
        lab = P.labels_for(P.conformal_sets(EPS), "worst"); truth = P.labels_for(None, "truth"); decl = P.labels_for(None, "declared")
        for k, task in enumerate(sample_tasks(seed, NT if os.environ.get('EMERKIT_GRID') == 'full' else {'B3': 72}.get(series, NT))):
            for name, K, W, dt in methods_for(METHODS[series], P, task, lab, decl, truth, deny, prm):
                rows.append(dict(series=series, config=ci, **cfg, seed=seed, task=k, type=task["type"], method=name, time=dt,
                                 **evaluate(P, task, K, W, deny, prm, truth)))
    return rows


def solver_scaling(series, ci, cfg):
    rows = []; prm = {**DEFAULTS, "N": cfg["N"], "k_max": cfg["k_max"], "T_max": cfg["T_max"]}; deny = norm_base()
    for seed in SEEDS:
        T, S = synth_catalogue(seed, providers=(12, 16)); P = Prepared(T, S, seed=seed)
        lab = P.labels_for(P.conformal_sets(EPS), "worst"); truth = P.labels_for(None, "truth")
        rng = np.random.default_rng(seed)
        for k, task in enumerate(sample_tasks(seed, 90)):
            pool = [c for c in TYPE_CAPS[task["type"]] if CAPS[c][1] is None or task["domain"] in CAPS[c][1]]
            task = dict(task, R=set(rng.choice(pool, min(cfg["nR"], len(pool)), replace=False)))
            r = compose_task(P, task, lab, deny, prm, explain=True)
            conflict = r["info"].get("conflict") if r["K"] is None else None
            rows.append(dict(series=series, config=ci, **cfg, seed=seed, task=k, status=r["status"], time=r["time"],
                             Nv=len(r.get("Nv") or []), conflict_size=len(conflict) if conflict else np.nan))
    return rows


def catalogue_size(series, ci, cfg):
    rows = []; deny = norm_base()
    for seed in SEEDS:
        T, S = synth_catalogue(seed); R, RS = real_catalogue(REAL, seed); R2, RS2 = toucan_catalogue(seed)
        pool, pool_s = R + R2, {**RS, **RS2}
        keep = np.random.default_rng(seed).permutation(len(pool))[:max(0, cfg["size"] - len(T))]
        R = [pool[i] for i in sorted(keep)]; RS = {k: v for k, v in pool_s.items() if k in {t["server"] for t in R}}
        t0 = time.time(); P = Prepared(T + R, {**S, **RS}, seed=seed); prep_t = time.time() - t0
        lab = P.labels_for(P.conformal_sets(EPS), "worst")
        for k, task in enumerate(sample_tasks(seed, 100)):
            t0 = time.time(); r = compose_task(P, task, lab, deny, DEFAULTS, explain=False); te = time.time() - t0
            t0 = time.time(); rag_mcp(P, task, 4); tr = time.time() - t0
            t0 = time.time(); zeng(P, task); tz = time.time() - t0
            rows.append(dict(series=series, config=ci, **cfg, n_tools=len(P.tools), seed=seed, task=k, prep_s=prep_t, emerkit_s=te,
                             rag_s=tr, zeng_s=tz, eq=int(r["K"] is not None), real_in_K=int(any(not P.tools[i]["synthetic"] for i in (r["K"] or []))),
                             maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    return rows


def end_to_end(series, ci, cfg):
    rows = []; deny = norm_base(cfg.get("profile", "standard")); prm = dict(DEFAULTS)
    for seed in SEEDS:
        T, S = synth_catalogue(seed); P = Prepared(T, S, seed=seed)
        lab = P.labels_for(P.conformal_sets(EPS), "worst"); truth = P.labels_for(None, "truth"); decl = P.labels_for(None, "declared")
        for qi, rq in enumerate(generate_requests(seed, n_per_domain=8)):
            rng = np.random.default_rng(seed * 1000 + qi)
            b = run_request(P, rq, lab, deny, prm, rng, n_W=3, s_max=cfg["s_max"], nu=cfg["nu"], mode=cfg["mode"], sub_density=cfg["density"])
            if b is None:                                  # no workflow passed the validator
                rows.append(dict(series=series, config=ci, **cfg, seed=seed, request=qi, domain=rq["domain"], method="EmerKit", n_tasks=0,
                                 equipped=0.0, substituted=0.0, gaps=1.0, dfr=0.0, any_dfr=0, backtracks=0, time=np.nan, all_equipped=0,
                                 obj_cov=0.0, prereq_viol=np.nan, preservation_viol=np.nan, no_workflow=1))
                continue
            n = len(b["wf"]["tasks"]); eqt = [p for p in b["plan"].values() if p["K"] is not None]
            dfr = [int(bool(risk(P.tools, p["K"], truth, p["task"], deny))) for p in eqt if p["K"]]
            rows.append(dict(series=series, config=ci, **cfg, seed=seed, request=qi, domain=rq["domain"], method="EmerKit", n_tasks=n,
                             equipped=len(eqt) / n, substituted=len(b["subs"]) / n, gaps=len(b["gaps"]) / n,
                             dfr=np.mean(dfr) if dfr else 0.0, any_dfr=int(any(dfr)), backtracks=b["backtracks"], time=b["time"],
                             all_equipped=int(len(b["gaps"]) == 0), **workflow_measures(b, rq["targets"], b["sub_cat"])))
            if ci == 0 and series == "C2":     # C1 without language models: same generator, no substitution, tools chosen by each baseline
                wfs = generate_candidates(np.random.default_rng(seed * 1000 + qi), rq, 1)
                if not wfs:
                    continue
                wf = wfs[0]
                for name in ["Gen+RAG-MCP", "Gen+RAG-MCP+filter", "Gen+Zeng", "Gen+Zeng+filter"]:
                    eqn, d = 0, []
                    for i in range(len(wf["tasks"])):
                        v = _task_view(wf, i, "isolated", {}, lab, P.tools)
                        K = (rag_mcp(P, v, 4, decl if "filter" in name else None, deny) if "RAG" in name
                             else zeng(P, v, prm, decl if "filter" in name else None, deny))
                        if K is not None and v["R"]:
                            e = evaluate(P, v, K, None, deny, prm, truth)
                            ok = e["cov"] == 1 and e["qviol"] == 0
                            eqn += ok; d.append(e["dfr"])
                    n = len(wf["tasks"])
                    rows.append(dict(series=series, config=ci, **cfg, seed=seed, request=qi, domain=rq["domain"], method=name, n_tasks=n,
                                     equipped=eqn / n, substituted=0.0, gaps=1 - eqn / n, dfr=np.mean(d) if d else 0.0, any_dfr=int(any(d)),
                                     backtracks=0, time=np.nan, all_equipped=int(eqn == n), obj_cov=1.0))
    return rows


def gap_map(series, ci, cfg):
    rows = []; prm = dict(DEFAULTS)
    for seed in SEEDS:
        T, S = synth_catalogue(seed)
        if series == "E1" and cfg["catalogue"] in ("mixed", "mcp-tools"):
            R, RS = real_catalogue(REAL, seed); T, S = T + R, {**S, **RS}
        if series == "E1" and cfg["catalogue"] == "toucan":
            R, RS = toucan_catalogue(seed); T, S = T + R, {**S, **RS}
        rng = np.random.default_rng(seed + 77)
        if series == "E2" and cfg["remove"] > 0:
            drop = set(rng.choice(sorted(S), int(cfg["remove"] * len(S)), replace=False))
            T = [t for t in T if t["server"] not in drop]; S = {k: v for k, v in S.items() if k not in drop}
        if series == "E2" and cfg["annotate"] > 0:
            T = copy.deepcopy(T)
            for t in T:
                if t["truth"] is not None and rng.random() < cfg["annotate"]:
                    t["hints"] = dict(t["truth"])
        P = Prepared(T, S, seed=seed); deny = norm_base(cfg.get("profile", "standard"))
        A = archetypes(seed)
        cls = classify_archetypes(P, A, deny, prm, P.conformal_sets(EPS))
        for k, (a, c) in enumerate(zip(A, cls)):
            rows.append(dict(series=series, config=ci, **cfg, seed=seed, archetype=k, type=a["type"], domain=a["domain"], dem=a["dem"], cls=c))
    return rows


RUN = {"P1": task_level, "P2": end_to_end, "A2": task_level, "A3": task_level, "A4": task_level, "B2": task_level, "B3": task_level, "D2": task_level,
       "C3": solver_scaling, "C4": catalogue_size, "C2": end_to_end, "E1": gap_map, "E2": gap_map}

if __name__ == "__main__":
    if sys.argv[1] == "list":
        for k, v in GRIDS.items():
            print(k, len(v))
        sys.exit()
    s = sys.argv[1]
    for c in sys.argv[2:]:
        t0 = time.time(); cfg = GRIDS[s][int(c)]
        rows = RUN[s](s, int(c), cfg)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{s}_c{c}.csv"), index=False)
        print(f"{s} c{c} {cfg} {time.time() - t0:.0f}s", flush=True)
