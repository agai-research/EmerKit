"""Property tests of the implementation against the definitions and propositions of the paper."""
import sys, os, itertools, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scipy.stats import binom
from emerkit.data import synth_catalogue, sample_tasks, generate_requests, LABELS
from emerkit.prep import Prepared
from emerkit.vocab import norm_base, CAPS
from emerkit.compose import compose_task, forbidden_sets, risk, zeng, rag_mcp, task_bounds, candidates_all, DEFAULTS, _build, explain_conflict, fit_scores, shortlist, qos_deficit, _feasible
from emerkit.e2e import run_request, preserves, sub_catalogue, validate, generate_candidates, run_workflow

T, S = synth_catalogue(1); P = Prepared(T, S, seed=1); deny = norm_base(); truth = P.labels_for(None, "truth")
tasks = sample_tasks(1, 120); prm = dict(DEFAULTS); rng = np.random.default_rng(0)
res = {}

def check(name, ok, detail=""):
    res[name] = ok; print(("PASS " if ok else "FAIL ") + name, detail)

# 1. risk == 0  <=>  no forbidden tool and no forbidden pair inside K (constraints encode the flow model exactly)
bad = 0
for task in tasks[:60]:
    C = candidates_all(P, task)
    F, Pp = forbidden_sets(T, C, truth, task, deny)
    for _ in range(20):
        K = list(rng.choice(C, min(len(C), int(rng.integers(1, 5))), replace=False)) if C else []
        cons = (not any(i in F for i in K)) and not any((i, j) in Pp for i in K for j in K if i != j)
        bad += int((risk(T, K, truth, task, deny) == 0) != cons)
check("flow constraints <=> risk model", bad == 0, f"mismatches={bad}")

# 2. soundness (Prop. 2): when every candidate's true label lies in its set, the worst-case toolkit has zero true risk
G = P.conformal_sets(0.1); lab = P.labels_for(G, "worst"); viol = n = 0
for task in tasks:
    C = candidates_all(P, task)
    if all(truth[i][l] in G[l][i] or i in P.cal_set for i in C for l in LABELS):
        r = compose_task(P, task, lab, deny, prm, explain=False)
        if r["K"]:
            n += 1; viol += int(risk(T, r["K"], truth, task, deny) > 0)
check("soundness of the worst-case check", viol == 0, f"covered tasks with toolkit={n}, violations={viol}")

# 3. every returned toolkit satisfies coverage, size, budget, QoS, hand-offs and flow constraints
errs = 0
for task in tasks:
    r = compose_task(P, task, lab, deny, prm, explain=False)
    if not r["K"]:
        continue
    K, W = r["K"], r["w"]
    ok = set(W) == task["R"] and all(W[x] in K and x in T[W[x]]["caps"] for x in W) and len(K) <= prm["k_max"]
    load = sum(T[i]["cost"] for i in K) + sum(S[s]["preamble"] for s in {T[i]["server"] for i in K}); ok &= load <= prm["B"]
    latb, prb, depb = task_bounds(P, task, candidates_all(P, task), prm["tau"])
    ok &= sum(T[i]["lat"] for i in W.values()) <= latb * 1.001 and sum(T[i]["price"] for i in W.values()) <= prb * 1.001 + 1e-6
    dep = math.prod(S[s]["avail"] for s in {T[i]["server"] for i in K}) * math.prod(T[i]["rel"] for i in W.values()); ok &= dep >= depb * 0.999
    ok &= all(any(a in T[i]["prod"] for i in K) for a in task.get("ho_out", set())) and all(any(a in T[i]["cons"] for i in K) for a in task.get("ho_in", set()))
    F, Pp = forbidden_sets(T, r["Nv"], lab, task, deny); ok &= not any(i in F for i in K) and not any((i, j) in Pp for i in K for j in K if i != j)
    errs += int(not ok)
check("returned toolkits satisfy all constraints", errs == 0, f"violations={errs}")

# 4. exactness: on small shortlists, brute force over every toolkit and serving assignment, with the same coefficients
worse = n = 0
for task in [t for t in tasks if 1 <= len(t["R"]) <= 2][:25]:
    p2 = {**prm, "N": 8, "n_min": 1}
    r = compose_task(P, task, lab, deny, p2, explain=False)
    if not r["K"]:
        continue
    n += 1; Nv = r["Nv"]; cands = candidates_all(P, task)
    g = fit_scores(P, task, cands, p2["eta"]); _, best_ = shortlist(P, task, cands, g, p2["N"], p2["n_min"])
    dq = qos_deficit(P, Nv); latb, prb, depb = task_bounds(P, task, cands, p2["tau"]); F, Pp = forbidden_sets(T, Nv, lab, task, deny)
    bmax = max(best_[i] for i in Nv) or 1.0; thB = int(math.floor(p2["theta"] * p2["B"])); Eth = max(1, p2["B"] - thB)
    top = -10 ** 9
    for k in range(1, p2["k_max"] + 1):
        for K in itertools.combinations(Nv, k):
            if any(i in F for i in K) or any((i, j) in Pp for i in K for j in K if i != j): continue
            if not all(any(a in T[i]["prod"] for i in K) for a in task.get("ho_out", set())): continue
            if not all(any(a in T[i]["cons"] for i in K) for a in task.get("ho_in", set())): continue
            servers = {T[i]["server"] for i in K}; L = sum(T[i]["cost"] for i in K) + sum(S[s_]["preamble"] for s_ in servers)
            if L > p2["B"]: continue
            pen = sum(int(1000 * p2["lam_I"] * (best_[i] / bmax) * (best_[j] / bmax) * P.interference(i, j)) for i, j in itertools.combinations(K, 2)
                      if P.interference(i, j) > 1e-3)
            pen += int(1000 * p2["lam_C"] * (max(0, L - thB) / Eth) ** 2)
            Rs = sorted(task["R"])
            for W in itertools.product(*[[i for i in K if x in T[i]["caps"]] for x in Rs]):
                if len(W) != len(Rs): continue
                if sum(int(T[i]["lat"]) for i in W) > int(latb) or sum(int(T[i]["price"] * 1e6) for i in W) > int(prb * 1e6): continue
                if sum(int(-math.log(S[s_]["avail"]) * 1e5) for s_ in servers) + sum(int(-math.log(T[i]["rel"]) * 1e5) for i in W) > int(-math.log(depb) * 1e5): continue
                u = sum(int(1000 * max(0.0, g[(i, x)] * (1 - p2["lam_Q"] * dq[i]))) for i, x in zip(W, Rs))
                top = max(top, u - pen)
    worse += int(top > round(r["obj"] * 1000) + 1)
check("composer is optimal on its shortlist (brute force)", worse == 0, f"instances={n}, better solutions found={worse}")

# 5. minimal conflicts: removing any group from the explanation makes the program feasible
nonmin = n = 0
for task in tasks:
    r = compose_task(P, task, lab, deny, prm, explain=True)
    if r["status"] != "infeasible":
        continue
    cands = candidates_all(P, task); g = fit_scores(P, task, cands, prm["eta"]); Nv, _ = shortlist(P, task, cands, g, prm["N"], prm["n_min"])
    dq = qos_deficit(P, Nv); b = task_bounds(P, task, cands, prm["tau"]); F, Pp = r["info"]["F"], r["info"]["P"]
    conf = set(r["info"]["conflict"]); n += 1
    ok = not _feasible(P, task, Nv, g, dq, b, F, Pp, prm, conf) and all(_feasible(P, task, Nv, g, dq, b, F, Pp, prm, conf - {x}) for x in conf)
    nonmin += int(not ok)
    if n >= 25: break
check("conflict explanations are minimal and infeasible", nonmin == 0, f"instances={n}, failures={nonmin}")

# 6. Zeng et al.: one provider per capability, bounds met
errs = n = 0
for task in tasks:
    K = zeng(P, task, prm)
    if K is None: continue
    n += 1; errs += int(not all(any(r in T[i]["caps"] for i in K) for r in task["R"]) or len(K) > len(task["R"]))
check("Zeng et al. baseline selects at most one service per capability and covers all", errs == 0, f"instances={n}, errors={errs}")

# 7. RAG-MCP + filter: no denied flow under declared labels
decl = P.labels_for(None, "declared"); errs = sum(int(risk(T, rag_mcp(P, t, 4, decl, deny), decl, t, deny) > 0) for t in tasks)
check("RAG-MCP filter removes every declared denied flow", errs == 0, f"errors={errs}")

# 8. substitutions satisfy the preservation relation; validated workflows pass the contract
n = bad = 0
for i, rq in enumerate(generate_requests(1, 3)):
    b = run_request(P, rq, lab, deny, prm, np.random.default_rng(i))
    if b is None: continue
    for sb in b["subs"]:
        n += 1; bad += int(not preserves(sb["original"], sb["substitute"], sub_catalogue("medium")))
    bad += len(validate(b["wf"]))
check("substitutions stay in the catalogue and workflows stay valid", bad == 0, f"substitutions={n}, errors={bad}")

# 9. Learn-then-Test acceptance rule
check("binomial acceptance rule (45 tasks suffice at xi=0.05, zeta=0.1)", binom.cdf(0, 45, 0.05) <= 0.1 < binom.cdf(0, 44, 0.05))
# 10. backtracking (conservative mode): an upstream tool that returns behaviour data blocks a public-web task downstream;
#     the procedure recomposes the upstream task without it and equips both tasks
from emerkit.vocab import ATT as _ATT
def _tool(i, cap, role, server, src_attrs, out, lat, desc_extra=""):
    tr = {f"src_{a}": int(a in src_attrs) for a in _ATT}; tr["out"] = out
    return dict(id=f"m{i}", name=f"tool{i}", role=role, server=server, desc=CAPS[cap][0] + desc_extra, params=["payload"], true_caps={cap},
                truth=tr, acc=set(_ATT), prod=set(), cons=set(), act=CAPS[cap][6], hints={l: tr[l] for l in LABELS}, lat=lat, price=0.0,
                rel=0.99, cost=60, synthetic=True, dup=None, domains={"programming", "datascience", "language"})
MT = [_tool(0, "flashcards", "INS", "ins-a", {"BEH"}, 0, 100), _tool(1, "flashcards", "INS", "ins-b", set(), 0, 220, " for revision"),
      _tool(2, "web_search", "PUB", "pub-w", set(), 1, 300), _tool(3, "explain_concept", "INS", "ins-c", set(), 0, 200)]
MS = {s_: dict(name=s_, role=r_, avail=0.99, preamble=40, synthetic=True) for s_, r_ in [("ins-a", "INS"), ("ins-b", "INS"), ("pub-w", "PUB"), ("ins-c", "INS")]}
MP = Prepared(MT, MS, seed=0); mtruth = MP.labels_for(None, "truth")
wf = dict(domain="programming", targets=["loops"], edges=[(0, 1, set())], tasks=[
    dict(type="practice", domain="programming", concepts=["loops"], level=3, R={"flashcards"}, D={"CONT"}, asm="none"),
    dict(type="acquisition", domain="programming", concepts=["loops"], level=2, R={"web_search"}, D={"PROF"}, asm="none")])
import copy
r_iso = run_workflow(MP, copy.deepcopy(wf), mtruth, deny, prm, rng, mode="isolated", use_sub=False)
r_con = run_workflow(MP, copy.deepcopy(wf), mtruth, deny, prm, rng, mode="conservative", use_sub=False)
first = [MT[i]["id"] for i in r_iso["plan"][0]["K"]]
check("backtracking recovers a downstream task in conservative mode", first == ["m0"] and r_con["backtracks"] == 1 and not r_con["gaps"]
      and [MT[i]["id"] for i in r_con["plan"][0]["K"]] == ["m1"], f"isolated upstream={first}, backtracks={r_con['backtracks']}, gaps={r_con['gaps']}")

print(f"\n{sum(res.values())}/{len(res)} properties hold")
sys.exit(0 if all(res.values()) else 1)
