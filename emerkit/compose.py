"""Flow check (Sect. 4.5), exact composition (Algorithm 2), calibration (Sect. 4.6) and the two LLM-free baselines."""
import math, time
import numpy as np
from scipy.stats import binom
from ortools.sat.python import cp_model
from .vocab import ATT, EXT_ROLES, CAPS, PHI
from .data import LABELS

SCALE = 1000
DEFAULTS = dict(B=2000, theta=0.75, k_max=4, N=20, n_min=2, T_max=10.0, lam_I=1.0, lam_C=1.0, lam_Q=0.5, eta=0.25,
                tau=1.25, beta=20.0, omega0=0.75, additive=False, use_qos=True, use_flow=True)


def src(lab):
    return {a for a in ATT if lab[f"src_{a}"]}


# ------------------------------------------------------------------------------------------------ flow model
def forbidden_sets(tools, cand, labels, task, deny):
    """F: tools forbidden alone; P: forbidden pairs (sink, source). Each with its causes (attribute, role)."""
    ty = task["type"]; A0 = set(task["D"]) | set(task.get("delta_in", set()))
    F, P = {}, {}
    for i in cand:
        t, lab = tools[i], labels[i]
        acc, role = t["acc"], t["role"]
        cause = {(a, role) for a in A0 & acc if deny[(ty, a, role)]}
        if lab["out"]:
            cause |= {(a, r) for a in (A0 | src(lab)) & acc for r in EXT_ROLES if deny[(ty, a, r)]}
        if cause:
            F[i] = cause
    for i in cand:
        if i in F:
            continue
        t, lab = tools[i], labels[i]
        for j in cand:
            if j == i or j in F:
                continue
            sj = src(labels[j]) & t["acc"]
            if not sj:
                continue
            cause = set()
            if tools[j]["server"] != t["server"]:
                cause |= {(a, t["role"]) for a in sj if deny[(ty, a, t["role"])]}
            if lab["out"]:
                cause |= {(a, r) for a in sj for r in EXT_ROLES if deny[(ty, a, r)]}
            if cause:
                P[(i, j)] = cause
    return F, P


def risk(tools, K, labels, task, deny):
    """Number of denied (attribute, role) flows that toolkit K makes possible; None if a label is unknown."""
    if any(labels[i] is None for i in K):
        return None
    F, P = forbidden_sets(tools, list(K), labels, task, deny)
    causes = set().union(*F.values()) if F else set()
    for v in P.values():
        causes |= v
    return len(causes)


# ------------------------------------------------------------------------------------------------ task-level quantities
def task_bounds(prep, task, cands_all, tau):
    """QoS bounds of a task: tau x sum of per-capability medians (latency, price); 1-(1-median dep) x tau."""
    T = prep.tools; S = prep.servers
    lat = pr = 0.0; dep = 1.0
    for r in task["R"]:
        prov = [i for i in cands_all if r in T[i]["caps"]]
        if not prov:
            continue
        lat += np.median([T[i]["lat"] for i in prov]); pr += np.median([T[i]["price"] for i in prov])
        dep *= np.median([T[i]["rel"] * S[T[i]["server"]]["avail"] for i in prov])
    return tau * lat, tau * pr + 1e-9, max(1e-6, 1 - (1 - dep) * tau)


def candidates_all(prep, task):
    return [i for i, t in enumerate(prep.tools) if t["caps"] & task["R"] and task["domain"] in t["domains"]]


def task_query(task):
    if task.get("query"):
        return task["query"]
    return task["type"] + " " + " ".join(CAPS[r][0] if r in CAPS else r for r in sorted(task["R"])) + " " + " ".join(c.replace("_", " ") for c in task["concepts"])


def fit_scores(prep, task, cands, eta):
    """g(t,r,v) = max(0, cos(e_t, e_{r,v})) * phi + eta * ev(t,v). When the task carries a natural-language request, e_{r,v}
    embeds the capability in the context of that request (capability text + request); otherwise the capability text alone."""
    T = prep.tools; g = {}
    ctx = {}
    if task.get("query") and cands:
        for r in task["R"]:
            j = prep.capnames.index(r)
            qv = prep.vec.transform([prep.capvec_text[j] + " " + task["query"]])
            ctx[r] = (prep.X[cands] @ qv.T).toarray().ravel()
    pos = {i: k for k, i in enumerate(cands)}
    for i in cands:
        for r in task["R"] & T[i]["caps"]:
            j = prep.capnames.index(r)
            cos = ctx[r][pos[i]] if r in ctx else prep.capsim[i, j]
            g[(i, r)] = max(0.0, cos) * PHI.get(T[i]["act"], {}).get(task["type"], 1.0) + eta * prep.evidence(i, task_query(task))
    return g


def qos_deficit(prep, cands):
    T, S = prep.tools, prep.servers
    if not cands:
        return {}
    def mm(v):
        v = np.asarray(v, float); lo, hi = v.min(), v.max()
        return np.zeros_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)
    lat = mm([T[i]["lat"] for i in cands]); pr = mm([T[i]["price"] for i in cands])
    av = mm([1 - S[T[i]["server"]]["avail"] for i in cands]); rel = mm([1 - T[i]["rel"] for i in cands])
    return {i: float((a + b + c + d) / 4) for i, a, b, c, d in zip(cands, lat, pr, av, rel)}


def shortlist(prep, task, cands_all, g, N, n_min):
    T = prep.tools
    best = {i: max([g.get((i, r), 0) for r in task["R"]] or [0]) for i in cands_all}
    order = sorted(cands_all, key=lambda i: -best[i])
    keep = []
    for r in task["R"]:
        keep += [i for i in sorted([i for i in cands_all if r in T[i]["caps"]], key=lambda i: -g.get((i, r), 0))][:n_min]
    for art in task.get("ho_out", set()):
        keep += [i for i in order if art in T[i]["prod"]][:n_min]
    for art in task.get("ho_in", set()):
        keep += [i for i in order if art in T[i]["cons"]][:n_min]
    for i in order:
        if len(set(keep)) >= max(N, len(set(keep))):
            break
        keep.append(i)
    return list(dict.fromkeys(keep)), best


# ------------------------------------------------------------------------------------------------ EmerKit composer
def _build(prep, task, Nv, g, dq, bounds, F, P, prm, groups):
    T, S = prep.tools, prep.servers
    m = cp_model.CpModel()
    x = {i: m.new_bool_var(f"x{i}") for i in Nv}
    w = {(r, i): m.new_bool_var(f"w{r}{i}") for r in task["R"] for i in Nv if r in T[i]["caps"]}
    servers = sorted({T[i]["server"] for i in Nv})
    z = {s: m.new_bool_var(f"z{k}") for k, s in enumerate(servers)}
    for (r, i), v in w.items():
        m.add(v <= x[i])
    for i in Nv:
        m.add(z[T[i]["server"]] >= x[i])
    for r in task["R"]:
        if ("cov", r) in groups:
            m.add(sum(v for (rr, i), v in w.items() if rr == r) == 1)
        else:
            m.add(sum(v for (rr, i), v in w.items() if rr == r) <= 1)
    if ("size",) in groups:
        m.add(sum(x.values()) <= prm["k_max"])
    if ("flow",) in groups:
        for i in F:
            m.add(x[i] == 0)
        for (i, j) in P:
            m.add(x[i] + x[j] <= 1)
    for art in task.get("ho_out", set()):
        if ("ho", art) in groups:
            m.add(sum(x[i] for i in Nv if art in T[i]["prod"]) >= 1)
    for art in task.get("ho_in", set()):
        if ("ho", art) in groups:
            m.add(sum(x[i] for i in Nv if art in T[i]["cons"]) >= 1)
    load = sum(T[i]["cost"] * x[i] for i in Nv) + sum(S[s]["preamble"] * z[s] for s in servers)
    if ("budget",) in groups:
        m.add(load <= prm["B"])
    if ("qos",) in groups and prm["use_qos"]:
        latb, prb, depb = bounds
        m.add(sum(int(T[i]["lat"]) * v for (r, i), v in w.items()) <= int(latb))
        m.add(sum(int(T[i]["price"] * 1e6) * v for (r, i), v in w.items()) <= int(prb * 1e6))
        m.add(sum(int(-math.log(S[s]["avail"]) * 1e5) * z[s] for s in servers)
              + sum(int(-math.log(T[i]["rel"]) * 1e5) * v for (r, i), v in w.items()) <= int(-math.log(depb) * 1e5))
    return m, x, w, z, load, servers


def compose_task(prep, task, labels, deny, prm=None, exclude=(), explain=True):
    """Returns dict(K, w, status, info). K is None when no admissible toolkit exists."""
    prm = {**DEFAULTS, **(prm or {})}
    t0 = time.time(); T = prep.tools
    if not task["R"]:
        return dict(K=[], w={}, status="no_tool_needed", info={}, time=0.0, obj=0.0)
    cands_all = [i for i in candidates_all(prep, task) if i not in exclude]
    unc = [r for r in task["R"] if not any(r in T[i]["caps"] for i in cands_all)]
    if unc:
        return dict(K=None, w=None, status="uncovered", info={"uncovered": unc}, time=time.time() - t0, obj=None)
    g = fit_scores(prep, task, cands_all, prm["eta"])
    Nv, best = shortlist(prep, task, cands_all, g, prm["N"], prm["n_min"])
    dq = qos_deficit(prep, Nv)
    bounds = task_bounds(prep, task, cands_all, prm["tau"])
    if prm["use_flow"]:
        F, P = forbidden_sets(T, Nv, labels, task, deny)
    else:
        F, P = {}, {}
    groups = {("cov", r) for r in task["R"]} | {("size",), ("flow",), ("budget",), ("qos",)} \
        | {("ho", a) for a in task.get("ho_out", set()) | task.get("ho_in", set())}
    m, x, w, z, load, servers = _build(prep, task, Nv, g, dq, bounds, F, P, prm, groups)
    # objective
    bmax = max(best[i] for i in Nv) or 1.0
    obj = []
    for (r, i), v in w.items():
        u = g[(i, r)] * (1 - prm["lam_Q"] * dq[i]) if prm["use_qos"] else g[(i, r)]
        obj.append(int(SCALE * max(0.0, u)) * v)
    if prm["additive"]:     # ablation: additive utility (every provider counts)
        obj = []
        for i in Nv:
            u = sum(g.get((i, r), 0) for r in task["R"]) * ((1 - prm["lam_Q"] * dq[i]) if prm["use_qos"] else 1)
            obj.append(int(SCALE * u) * x[i])
    if prm["lam_I"] > 0:
        for a in range(len(Nv)):
            for b in range(a + 1, len(Nv)):
                i, j = Nv[a], Nv[b]
                io = prep.interference(i, j, prm["beta"], prm["omega0"])
                if io <= 1e-3:
                    continue
                iv = (best[i] / bmax) * (best[j] / bmax) * io
                y = m.new_bool_var(f"y{i}_{j}")
                m.add(y >= x[i] + x[j] - 1)
                obj.append(-int(SCALE * prm["lam_I"] * iv) * y)
    if prm["lam_C"] > 0:
        thB = int(math.floor(prm["theta"] * prm["B"])); Eth = max(1, prm["B"] - thB)
        e = m.new_int_var(0, Eth, "e"); Lv = m.new_int_var(0, 10 ** 7, "L")
        m.add(Lv == load); m.add(e >= Lv - thB)
        top = int(SCALE * prm["lam_C"]) + 1
        sat = m.new_int_var(0, top, "S")
        m.add_element(e, [int(SCALE * prm["lam_C"] * (k / Eth) ** 2) for k in range(Eth + 1)], sat)
        obj.append(-1 * sat)
    m.maximize(sum(obj))
    sol = cp_model.CpSolver(); sol.parameters.max_time_in_seconds = prm["T_max"]; sol.parameters.num_workers = 1
    st = sol.solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        K = [i for i in Nv if sol.value(x[i])]
        W = {r: i for (r, i), v in w.items() if sol.value(v)}
        return dict(K=K, w=W, status="optimal" if st == cp_model.OPTIMAL else "feasible", info={"F": len(F), "P": len(P)},
                    time=time.time() - t0, Nv=Nv, obj=sol.objective_value / SCALE)
    status = "timeout" if st == cp_model.UNKNOWN else "infeasible"
    info = {"F": F, "P": P}
    if explain and status == "infeasible":
        info["conflict"] = explain_conflict(prep, task, Nv, g, dq, bounds, F, P, prm, groups)
    return dict(K=None, w=None, status=status, info=info, time=time.time() - t0, Nv=Nv, obj=None)


def _feasible(prep, task, Nv, g, dq, bounds, F, P, prm, groups):
    m, *_ = _build(prep, task, Nv, g, dq, bounds, F, P, prm, groups)
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = prm["T_max"]; s.parameters.num_workers = 1
    return s.solve(m) in (cp_model.OPTIMAL, cp_model.FEASIBLE)


def explain_conflict(prep, task, Nv, g, dq, bounds, F, P, prm, groups):
    """Deletion filter over constraint groups, in a fixed priority order; returns a minimal infeasible subset."""
    order = sorted(groups, key=lambda gr: ["budget", "qos", "size", "ho", "flow", "cov"].index(gr[0]))
    keep = set(groups)
    for gr in order:
        trial = keep - {gr}
        if not _feasible(prep, task, Nv, g, dq, bounds, F, P, prm, trial):
            keep = trial
    return sorted(keep, key=str)


# ------------------------------------------------------------------------------------------------ calibration (Learn then Test)
def calibrate(prep, cal_tasks, deny, xi=0.05, zeta=0.1, grid=None, prm=None):
    grid = grid if grid is not None else [round(0.02 * k, 2) for k in range(16)]
    truth = prep.labels_for(None, "truth")
    eps_hat, trace = 0.0, []
    for eps in grid:                                  # most conservative level first, fixed-sequence testing
        G = prep.conformal_sets(eps)
        lab = prep.labels_for(G, "worst")
        viol = ret = 0
        for task in cal_tasks:
            res = compose_task(prep, task, lab, deny, prm, explain=False)
            if res["K"]:
                ret += 1
                rk = risk(prep.tools, res["K"], truth, task, deny)
                viol += int(bool(rk))
        p = binom.cdf(viol, len(cal_tasks), xi)
        trace.append(dict(eps=eps, viol=viol, returned=ret, p=float(p)))
        if p <= zeta:
            eps_hat = eps
        else:
            break
    return eps_hat, trace


# ------------------------------------------------------------------------------------------------ baselines
def rag_mcp(prep, task, k, labels_declared=None, deny=None):
    """RAG-MCP: top-k tools by similarity between the task query and tool definitions; optional declared-annotation filter."""
    qv = prep.vec.transform([task_query(task)])
    sc = (prep.X @ qv.T).toarray().ravel()
    order = [i for i in np.argsort(-sc) if task["domain"] in prep.tools[i]["domains"]]
    if labels_declared is None:
        return order[:k]
    K = []
    for i in order:
        if risk(prep.tools, K + [i], labels_declared, task, deny) == 0:
            K.append(i)
        if len(K) == k:
            break
    return K


def zeng(prep, task, prm=None, labels_declared=None, deny=None):
    """Zeng et al.: one service per abstract task (capability), maximize SAW QoS utility under global QoS constraints (integer program)."""
    prm = {**DEFAULTS, **(prm or {})}
    T, S = prep.tools, prep.servers
    cands = candidates_all(prep, task)
    if not task["R"] or not cands or any(not any(r in T[i]["caps"] for i in cands) for r in task["R"]):
        return None
    dq = qos_deficit(prep, cands)
    latb, prb, depb = task_bounds(prep, task, cands, prm["tau"])
    m = cp_model.CpModel()
    w = {(r, i): m.new_bool_var("") for r in task["R"] for i in cands if r in T[i]["caps"]}
    x = {i: m.new_bool_var("") for i in cands}
    for (r, i), v in w.items():
        m.add(v <= x[i])
    for r in task["R"]:
        m.add(sum(v for (rr, i), v in w.items() if rr == r) == 1)
    for i in cands:
        m.add(x[i] <= sum(v for (r, j), v in w.items() if j == i))
    m.add(sum(int(T[i]["lat"]) * v for (r, i), v in w.items()) <= int(latb))
    m.add(sum(int(T[i]["price"] * 1e6) * v for (r, i), v in w.items()) <= int(prb * 1e6))
    m.add(sum(int(-math.log(T[i]["rel"] * S[T[i]["server"]]["avail"]) * 1e5) * v for (r, i), v in w.items()) <= int(-math.log(depb) * 1e5))
    if labels_declared is not None:
        F, P = forbidden_sets(T, cands, labels_declared, task, deny)
        for i in F:
            m.add(x[i] == 0)
        for (i, j) in P:
            m.add(x[i] + x[j] <= 1)
    m.maximize(sum(int(SCALE * (1 - dq[i])) * v for (r, i), v in w.items()))
    s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = prm["T_max"]; s.parameters.num_workers = 1
    st = s.solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [i for i in cands if s.value(x[i])]


# ---------------------------------------------------------------------------------------------- greedy composer (comparison)
def greedy_task(prep, task, labels, deny, prm=None):
    """Greedy counterpart of compose_task: same shortlist, coefficients, and constraints; capabilities are served one at a time
    (scarcest first) by the provider with the best marginal objective. Used to measure the value of exact composition."""
    prm = {**DEFAULTS, **(prm or {})}; t0 = time.time(); T, S = prep.tools, prep.servers
    cands_all = candidates_all(prep, task)
    if not task["R"] or any(not any(r in T[i]["caps"] for i in cands_all) for r in task["R"]):
        return dict(K=None, w=None, obj=None, time=time.time() - t0, status="uncovered")
    g = fit_scores(prep, task, cands_all, prm["eta"]); Nv, best = shortlist(prep, task, cands_all, g, prm["N"], prm["n_min"])
    dq = qos_deficit(prep, Nv); latb, prb, depb = task_bounds(prep, task, cands_all, prm["tau"])
    F, P = forbidden_sets(T, Nv, labels, task, deny) if prm["use_flow"] else ({}, {})
    bmax = max(best[i] for i in Nv) or 1.0; thB = int(math.floor(prm["theta"] * prm["B"])); Eth = max(1, prm["B"] - thB)

    def value(K, W):
        u = sum(int(SCALE * max(0.0, g[(i, r)] * (1 - prm["lam_Q"] * dq[i]))) for r, i in W.items())
        pen = 0
        for a in range(len(K)):
            for b in range(a + 1, len(K)):
                io = prep.interference(K[a], K[b], prm["beta"], prm["omega0"])
                if io > 1e-3:
                    pen += int(SCALE * prm["lam_I"] * (best[K[a]] / bmax) * (best[K[b]] / bmax) * io)
        L = sum(T[i]["cost"] for i in K) + sum(S[s]["preamble"] for s in {T[i]["server"] for i in K})
        return u - pen - int(SCALE * prm["lam_C"] * (max(0, L - thB) / Eth) ** 2), L

    def feasible(K, W):
        if len(K) > prm["k_max"] or any(i in F for i in K) or any((i, j) in P for i in K for j in K if i != j):
            return False
        if value(K, W)[1] > prm["B"]:
            return False
        if sum(int(T[i]["lat"]) for i in W.values()) > int(latb) or sum(int(T[i]["price"] * 1e6) for i in W.values()) > int(prb * 1e6):
            return False
        dep = sum(int(-math.log(S[s]["avail"]) * 1e5) for s in {T[i]["server"] for i in K}) + sum(int(-math.log(T[i]["rel"]) * 1e5) for i in W.values())
        return dep <= int(-math.log(depb) * 1e5)

    K, W = [], {}
    for r in sorted(task["R"], key=lambda r: sum(1 for i in Nv if r in T[i]["caps"])):
        top = None
        for i in Nv:
            if r not in T[i]["caps"]:
                continue
            K2 = K if i in K else K + [i]; W2 = {**W, r: i}
            if feasible(K2, W2):
                v = value(K2, W2)[0]
                if top is None or v > top[0]:
                    top = (v, K2, W2)
        if top is None:
            return dict(K=None, w=None, obj=None, time=time.time() - t0, status="greedy_fail")
        _, K, W = top
    for a in task.get("ho_out", set()) | task.get("ho_in", set()):        # hand-offs: add the cheapest tool that provides the artefact
        key = "prod" if a in task.get("ho_out", set()) else "cons"
        if any(a in T[i][key] for i in K):
            continue
        opts = [K + [i] for i in Nv if i not in K and a in T[i][key] and feasible(K + [i], W)]
        if not opts:
            return dict(K=None, w=None, obj=None, time=time.time() - t0, status="greedy_fail")
        K = max(opts, key=lambda k: value(k, W)[0])
    return dict(K=K, w=W, obj=value(K, W)[0] / SCALE, time=time.time() - t0, status="greedy")
