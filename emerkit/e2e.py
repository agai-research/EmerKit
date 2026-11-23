"""End-to-end procedure (Algorithm 3): validated workflows, per-task composition, backtracking (conservative mode),
objective-preserving substitution, gap records; and the ecosystem gap map (Sect. 4.9)."""
import time
import numpy as np
from .vocab import TYPES, LEVEL, CAPS, TYPE_CAPS, SUB, HANDOFF_ATTRS, DOMAIN_CONCEPTS, DEFAULT_D
from .data import generate_workflow, make_task, inherited_attrs
from .compose import compose_task, risk, src, candidates_all


# ------------------------------------------------------------------------------------------------ generator contract
def validate(wf):
    """Contract (a)-(d): acyclic chain, prerequisites upstream, admissible capabilities, typed hand-offs; D(v) completed."""
    pre = dict(DOMAIN_CONCEPTS[wf["domain"]]); errors = []
    first = {}
    for i, t in enumerate(wf["tasks"]):
        for c in t["concepts"]:
            first.setdefault(c, i)
    for i, t in enumerate(wf["tasks"]):
        for c in t["concepts"]:
            if any(p in first and first[p] > first[c] for p in pre[c]):
                errors.append(("prerequisite", i, c))
        bad = [r for r in t["R"] if r not in TYPE_CAPS[t["type"]] or (CAPS[r][1] and wf["domain"] not in CAPS[r][1])]
        if bad:
            errors.append(("capability", i, bad))
        t["D"] = set(t["D"]) | DEFAULT_D[t["type"]]
    for (a, b, ho) in wf["edges"]:
        if not (a < b) or any(h not in HANDOFF_ATTRS for h in ho):
            errors.append(("edge", a, b))
    covered = {c for t in wf["tasks"] for c in t["concepts"]}
    if not set(wf["targets"]) <= covered:
        errors.append(("objectives", wf["targets"]))
    return errors


def repair_workflow(wf):
    """Deterministic repair of a generated workflow before validation: keep concepts of the referential, keep only capabilities
    admissible for the type and domain (adding the core capability when none is left), order tasks by the prerequisite order of
    their concepts, and add a practice task for every target concept left uncovered. Repairs are counted and reported."""
    from .vocab import CORE_CAP
    order = [c for c, _ in DOMAIN_CONCEPTS[wf["domain"]]]; rank = {c: k for k, c in enumerate(order)}
    fixes = 0; tasks = []
    for t in wf["tasks"]:
        t = dict(t)
        cs = [c for c in t["concepts"] if c in rank]
        fixes += len(cs) != len(t["concepts"])
        if not cs:
            continue
        ok = {r for r in t["R"] if r in TYPE_CAPS[t["type"]] and (CAPS[r][1] is None or wf["domain"] in CAPS[r][1])}
        if not ok:
            core = CORE_CAP.get((t["type"], wf["domain"])) or CORE_CAP.get((t["type"], None))
            ok = {core} if core else set()
        fixes += ok != set(t["R"])
        t.update(concepts=cs, R=ok, D=set(t["D"]) | DEFAULT_D[t["type"]]); tasks.append(t)
    covered = {c for t in tasks for c in t["concepts"]}
    for c in wf["targets"]:
        if c not in covered and c in rank:
            core = CORE_CAP.get(("practice", wf["domain"]))
            tasks.append(dict(type="practice", domain=wf["domain"], concepts=[c], level=LEVEL["practice"], R={core}, D=set(DEFAULT_D["practice"]), asm="none")); fixes += 1
    new = sorted(tasks, key=lambda t: (min(rank[c] for c in t["concepts"]), TYPES.index(t["type"])))
    fixes += [id(t) for t in new] != [id(t) for t in tasks]
    return dict(wf, tasks=new, edges=[(i, i + 1, set()) for i in range(len(new) - 1)], repairs=fixes)


def generate_candidates(rng, rq, n_W):
    out = []
    for _ in range(n_W):
        for _ in range(3):                                    # at most two regenerations
            wf = generate_workflow(rng, rq["domain"], rq["targets"], rq["known"])
            if not validate(wf):
                out.append(wf); break
    return out


def _task_view(wf, i, mode, plan, labels, tools):
    t = dict(wf["tasks"][i])
    t["ho_out"] = set().union(*[ho for (a, b, ho) in wf["edges"] if a == i] or [set()])
    t["ho_in"] = set().union(*[ho for (a, b, ho) in wf["edges"] if b == i] or [set()])
    t["delta_in"] = inherited_attrs(wf, i)
    if mode == "conservative":                                # shared context: everything upstream may be present
        for u in range(i):
            t["delta_in"] |= set(wf["tasks"][u]["D"])
            for j in plan.get(u, {}).get("K", []) or []:
                t["delta_in"] |= src(labels[j])
    return t


# ------------------------------------------------------------------------------------------------ substitution
def preserves(v, vp, sub_cat):
    """Admissible substitute (Definition 9): same concepts, level not lower, type pair in Sub, hand-offs kept, not summative."""
    return (set(v["concepts"]) <= set(vp["concepts"]) and LEVEL[vp["type"]] >= LEVEL[v["type"]] and vp["type"] in sub_cat.get(v["type"], [])
            and v["asm"] != "summative" and all(any(a in CAPS[r][4] for r in vp["R"]) for a in v.get("ho_out", set()))
            and all(any(a in CAPS[r][5] for r in vp["R"]) for a in v.get("ho_in", set())))


def propose_substitutes(rng, v, sub_cat, n_s=4):
    out = []
    for ty in sub_cat.get(v["type"], []):
        for _ in range(n_s):
            vp = make_task(rng, v["domain"], ty, v["concepts"])
            for a in v.get("ho_out", set()):                  # keep the artefacts the task hands off
                prods = [r for r in TYPE_CAPS[ty] if a in CAPS[r][4] and (CAPS[r][1] is None or v["domain"] in CAPS[r][1])]
                if prods: vp["R"].add(prods[0])
            for a in v.get("ho_in", set()):
                cons = [r for r in TYPE_CAPS[ty] if a in CAPS[r][5] and (CAPS[r][1] is None or v["domain"] in CAPS[r][1])]
                if cons: vp["R"].add(cons[0])
            vp["ho_out"], vp["ho_in"], vp["delta_in"] = v.get("ho_out", set()), v.get("ho_in", set()), v.get("delta_in", set())
            out.append(vp)
    return out


def sub_catalogue(density):
    if density == "medium":
        return SUB
    if density == "panel":                                    # majority of the LLM expert panel; summative never automatic
        from .vocab import panel_majority
        maj = panel_majority("sub")
        return {a: [b for b in TYPES if maj.get((a, b), b in SUB.get(a, [])) and a != "summative"] for a in TYPES}
    if density == "low":
        return {k: [x for x in v if x == k] for k, v in SUB.items()}
    dense = {k: list(v) for k, v in SUB.items()}               # high: also cross-practice substitutions
    dense["formative"] = ["formative", "practice"]; dense["acquisition"] = ["acquisition", "discussion", "practice"]
    return dense


# ------------------------------------------------------------------------------------------------ Algorithm 3
def run_workflow(prep, wf, labels, deny, prm, rng, s_max=2, mode="isolated", sub_cat=None, use_sub=True):
    """Algorithm 3 on one validated workflow: per-task composition in topological order, one backtracking step in
    conservative mode, best admissible substitute, gap records. Returns the plan and its diagnostics."""
    sub_cat = sub_cat or SUB
    plan, gaps, subs, causes, backtracks = {}, [], [], [], 0
    for i in range(len(wf["tasks"])):
        v = _task_view(wf, i, mode, plan, labels, prep.tools)
        res = compose_task(prep, v, labels, deny, prm)
        if res["K"] is None and mode == "conservative" and i > 0:
            extra = v["delta_in"] - set(v["D"]) - inherited_attrs(wf, i)
            for u in range(i - 1, -1, -1):
                bad = [j for j in (plan.get(u, {}).get("K") or []) if src(labels[j]) & extra]
                if not bad:
                    continue
                uv = _task_view(wf, u, mode, plan, labels, prep.tools)
                ru = compose_task(prep, uv, labels, deny, prm, exclude=set(bad), explain=False)
                if ru["K"] is not None:
                    trial = dict(plan); trial[u] = dict(ru, task=uv)
                    v2 = _task_view(wf, i, mode, trial, labels, prep.tools)
                    r2 = compose_task(prep, v2, labels, deny, prm)
                    if r2["K"] is not None:
                        plan, v, res = trial, v2, r2; backtracks += 1
                break
        if res["K"] is None and use_sub and len(subs) < s_max and v["asm"] != "summative":
            cand = None
            for vp in propose_substitutes(rng, v, sub_cat):
                if not preserves(v, vp, sub_cat):
                    continue
                rp = compose_task(prep, vp, labels, deny, prm, explain=False)
                if rp["K"] is not None and (cand is None or rp["obj"] > cand[1]["obj"]):
                    cand = (vp, rp)
            if cand is not None:
                subs.append(dict(index=i, original=dict(v), substitute=dict(cand[0])))
                v, res = cand
                wf["tasks"][i] = {k: cand[0][k] for k in ("type", "domain", "concepts", "level", "R", "D", "asm")}
        if res["K"] is None:
            gaps.append(i); causes.append(res["status"])
        plan[i] = dict(res, task=v)
    return dict(wf=wf, plan=plan, gaps=gaps, subs=subs, causes=causes, backtracks=backtracks)


def run_request(prep, rq, labels, deny, prm, rng, n_W=3, s_max=2, mu=1.0, nu=0.5, mode="isolated", sub_density="medium",
                use_sub=True):
    t0 = time.time(); sub_cat = sub_catalogue(sub_density)
    best, best_val = None, -np.inf
    for wf in generate_candidates(rng, rq, n_W):
        r = run_workflow(prep, wf, labels, deny, prm, rng, s_max, mode, sub_cat, use_sub)
        val = sum(p["obj"] for p in r["plan"].values() if p["K"] is not None) - mu * len(r["gaps"]) - nu * len(r["subs"])
        if val > best_val:
            best_val, best = val, r
    if best is not None:
        best["time"] = time.time() - t0
        best["sub_cat"] = sub_cat
    return best


def workflow_measures(best, targets, sub_cat):
    """CovObj (targets covered by equipped tasks), prerequisite violations of the final workflow, preservation violations."""
    wf, plan = best["wf"], best["plan"]
    covered = {c for i, t in enumerate(wf["tasks"]) if plan[i]["K"] is not None for c in t["concepts"]}
    prereq = sum(1 for e in validate(wf) if e[0] == "prerequisite")
    pres = sum(1 for s in best["subs"] if not preserves(s["original"], s["substitute"], sub_cat))
    return dict(obj_cov=len(set(targets) & covered) / len(set(targets)), prereq_viol=prereq, preservation_viol=pres)


# ------------------------------------------------------------------------------------------------ gap map
def classify_archetypes(prep, archs, deny, prm, G):
    """UNCOVERED / RISKY / UNCERTAIN / SAFE for each archetype (best-case labels, then worst-case labels)."""
    lab_best = prep.labels_for(G, "best"); lab_worst = prep.labels_for(G, "worst")
    out = []
    for a in archs:
        cands = candidates_all(prep, a)
        if any(not any(r in prep.tools[i]["caps"] for i in cands) for r in a["R"]):
            out.append("UNCOVERED"); continue
        if compose_task(prep, a, lab_best, deny, prm, explain=False)["K"] is None:
            out.append("RISKY"); continue
        out.append("SAFE" if compose_task(prep, a, lab_worst, deny, prm, explain=False)["K"] is not None else "UNCERTAIN")
    return out
