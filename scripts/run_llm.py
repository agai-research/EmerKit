"""Model-dependent series (run in Claude Code with API keys). Every model call is logged by emerkit.llm.LLMClient.
Usage:
  python scripts/run_llm.py probes-design SEED            # no model needed: writes data/probes/probes_seedS.jsonl
  python scripts/run_llm.py capabilities SEED             # A0 with an LLM extractor (synthetic truth + audited real tools)
  python scripts/run_llm.py probes SEED [MAX]             # A5: run the probes with $EMERKIT_MODEL; then `fit-probes`
  python scripts/run_llm.py fit-probes                    # A5: mixed-effects logistic fit over all models -> config/fitted_weights.json
  python scripts/run_llm.py a6 SEED [MAX]                 # A6: probe the toolkits of EmerKit, RAG-MCP and Zeng et al.
  python scripts/run_llm.py toucan [N]                    # A7 data: Toucan trajectories -> data/traces/toucan_traces.jsonl
  python scripts/run_llm.py toucan-caps                   # A7: LLM tagging of the Toucan tools with Toucan categories
  python scripts/run_llm.py a7 SEED                       # A7: toolkits vs tools of held-out trajectories (split by server)
  python scripts/run_llm.py c1 SEED                       # C1: Compositio Prompto and LLM generator + EmerKit
  python scripts/run_llm.py judges                        # LLM expert panel on pedagogical parameters and substitutions
  python scripts/run_llm.py audit-review                  # LLM review of the code of sampled real tools (audit step 2)
"""
import sys, os, json, math, time, re
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from emerkit.data import synth_catalogue, sample_tasks, generate_requests
from emerkit.prep import Prepared
from emerkit.vocab import norm_base, CAPS, PHI, SUB, TYPE_CAPS, DEFAULT_D, TYPES
from emerkit.compose import compose_task, rag_mcp, zeng, DEFAULTS, task_query, risk
from emerkit.metrics import evaluate
from emerkit import llm

OUT = os.path.join(ROOT, "results", "llm"); os.makedirs(OUT, exist_ok=True)
MODEL = os.environ.get("EMERKIT_MODEL", "unset").replace("/", "_")


def prepared(seed):
    T, S = synth_catalogue(seed); P = Prepared(T, S, seed=seed); return T, S, P


def sat_term(load, B=2000, theta=0.75):
    return max(0.0, (load - theta * B) / ((1 - theta) * B)) ** 2


def requests_bank():
    """Natural requests per capability, written by a model that is not the one probed (avoids literal matches with descriptions)."""
    client = llm.LLMClient(); path = os.path.join(ROOT, "data", "probes", "requests_bank.json")
    def run(r):
        q = (f"Write 6 short and varied requests that a learner or a teacher could type to an AI assistant and that require this "
             f"capability: '{CAPS[r][0]}'. Do not reuse its key words; make them concrete (a topic, a file, a goal). Return only a JSON list of strings.")
        return r, [x for x in (llm.parse_json(client("Return only JSON.", q, max_tokens=600)) or []) if isinstance(x, str)]
    bank = dict(llm.pmap(run, list(CAPS)))
    json.dump(bank, open(path, "w"), indent=1); print({k: len(v) for k, v in bank.items()})


def probes_design(seed):
    T, S, P = prepared(seed)
    d = llm.probe_design(np.random.default_rng(seed), P, ks=(2, 4, 8, 16, 32))
    bp = os.path.join(ROOT, "data", "probes", "requests_bank.json"); bank = json.load(open(bp)) if os.path.exists(bp) else {}
    path = os.path.join(ROOT, "data", "probes", f"probes_seed{seed}.jsonl"); os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for k, x in enumerate(d):
            reqs = bank.get(x["cap"]) or [f"{CAPS[x['cap']][0]} for a learner"]
            x = dict(x, probe=k, split="fit" if k % 2 == 0 else "test", task_text=reqs[k % len(reqs)],
                     interference=sum(P.interference(i, j) for a, i in enumerate(x["toolkit"]) for j in x["toolkit"][a + 1:]),
                     saturation=sat_term(x["load_ratio"] * 2000))
            f.write(json.dumps(x, default=int) + "\n")
    print(len(d), "probes ->", path)


def probes(seed, maxn=None):
    T, S, P = prepared(seed); client = llm.LLMClient()
    path = os.path.join(ROOT, "data", "probes", f"probes_seed{seed}.jsonl")
    allx = [json.loads(l) for l in open(path)]
    if maxn:                                                     # stratified subset: every cell keeps the same share
        stride = max(1, len(allx) // int(maxn)); allx = allx[::stride][: int(maxn)]
    def run(x):
        pad = ("Reference documentation. " * max(0, x["pad_tokens"] // 3))[: 4 * x["pad_tokens"]]
        chosen = llm.probe(client, x["task_text"] + ("\n" + pad if pad else ""), x["toolkit"], T)
        return dict(x, model=MODEL, chosen=chosen, correct=int(chosen is not None and x["cap"] in (T[chosen]["true_caps"] or set())))
    rows = llm.pmap(run, allx)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"A5_{MODEL}_seed{seed}.csv"), index=False)


def fit_probes():
    from emerkit.stats import fit_probe_model
    import glob
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "A5_*.csv"))])
    fit = fit_probe_model(d[d.split == "fit"])
    json.dump(fit, open(os.path.join(ROOT, "config", "fitted_weights.json"), "w"), indent=1)
    print(fit)


def a6(seed, maxn=None):
    """Probe the toolkits each method returns: for each task and required capability, does the agent pick a true provider?"""
    T, S, P = prepared(seed); client = llm.LLMClient(); deny = norm_base()
    lab = P.labels_for(P.conformal_sets(0.05), "worst"); jobs = []
    tasks = sample_tasks(seed, 300); stride = max(1, len(tasks) // int(maxn)) if maxn else 1
    for k, task in list(enumerate(tasks))[::stride][: int(maxn) if maxn else None]:
        kits = {"EmerKit": compose_task(P, task, lab, deny, DEFAULTS, explain=False)["K"], "RAG-MCP": rag_mcp(P, task, 4), "Zeng": zeng(P, task)}
        for m, K in kits.items():
            if not K: continue
            for r in sorted(task["R"]):
                jobs.append((k, task, m, K, r))
    def run(j):
        k, task, m, K, r = j
        c = llm.probe(client, f"{CAPS[r][0]} ({task['type']} on {', '.join(task['concepts'])})", K, T)
        return dict(seed=seed, task=k, method=m, cap=r, model=MODEL, size=len(K), correct=int(c is not None and r in (T[c]["true_caps"] or set())))
    rows = llm.pmap(run, jobs)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"A6_{MODEL}_seed{seed}.csv"), index=False)


def capabilities(seed):
    T, S, P = prepared(seed); client = llm.LLMClient()
    rows = llm.pmap(lambda t: dict(tool=t["id"], llm_caps=sorted(llm.llm_capabilities(client, t)), true_caps=sorted(t["true_caps"]),
                                   mapper_caps=sorted(t["caps"])), T)
    pd.DataFrame(rows).to_json(os.path.join(OUT, f"A0_{MODEL}_seed{seed}.json"), orient="records")


def toucan(n=20000):
    """Download a random sample of Toucan trajectories and keep, for each, the query and the (server, tool) pairs that were called."""
    rows = llm.toucan_sample(n=int(n))
    path = os.path.join(ROOT, "data", "traces", "toucan_traces.jsonl"); os.makedirs(os.path.dirname(path), exist_ok=True)
    print("columns:", list(rows[0].keys()))                       # the parser below is checked against these columns on first run
    with open(path, "w") as f:
        for r in rows:
            msgs = r.get("messages") or r.get("conversation") or []
            if isinstance(msgs, str):
                try: msgs = json.loads(msgs)
                except ValueError: msgs = []
            query = next((m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "user"), r.get("question", ""))
            called = []
            for m in msgs:
                if not isinstance(m, dict): continue
                for c in m.get("tool_calls") or m.get("function_call") and [m["function_call"]] or []:
                    fn = c.get("function", c); called.append(fn.get("name"))
            f.write(json.dumps(dict(query=str(query)[:2000], tools=[c for c in called if c], target_tools=r.get("target_tools"),
                                    servers=r.get("mcp_servers") or (r.get("metadata") or {}).get("mcp_servers"), success=True)) + "\n")


def a7(seed, maxn=500):
    """A7 on the Toucan catalogue: capability vocabulary = Toucan categories; tool and task capabilities from an LLM when the cache
    exists (data/traces/toucan_caps_<model>.json), else from the similarity mapper (declared). Norms are not involved (H1 only)."""
    from collections import defaultdict
    from emerkit.sources import toucan_catalogue
    T, S = toucan_catalogue(seed)
    cats = sorted({t["category"] for t in T if t.get("category")})
    vocab = {c.lower().replace(" & ", "_").replace(" ", "_").replace("/", "_"): c for c in cats}
    cache = os.path.join(ROOT, "data", "traces", f"toucan_caps_{MODEL}.json")
    keep = False
    if os.path.exists(cache):
        caps = json.load(open(cache))
        for t in T:
            t["caps"] = set(caps.get(f"{t['server']}::{t['name']}", [])); keep = True
    name_idx = {}
    for i, t in enumerate(T):
        name_idx.setdefault(t["name"], []).append(i)
    traces = [json.loads(l) for l in open(os.path.join(ROOT, "data", "traces", "toucan_traces.jsonl"))]
    rng = np.random.default_rng(seed); servers = sorted(S); rng.shuffle(servers); test_srv = set(servers[: int(0.3 * len(servers))])
    def idx(tr): return [i for n in tr["tools"] for i in name_idx.get(n, [])[:1]]
    fit_tr = [dict(tr, tools=idx(tr)) for tr in traces if idx(tr) and not any(T[i]["server"] in test_srv for i in idx(tr))]
    test_tr = [dict(tr, tools=idx(tr)) for tr in traces if idx(tr) and all(T[i]["server"] in test_srv for i in idx(tr))]
    P = Prepared(T, S, seed=seed, cap_vocab=vocab, keep_caps=keep); P.set_traces(fit_tr)
    deny = defaultdict(bool); labels = [{l: 0 for l in ["src_ID", "src_PROF", "src_PERF", "src_BEH", "src_CONT", "out"]}] * len(T)
    rows = []
    for k, tr in enumerate(test_tr[: int(maxn)]):
        truth_set = set(tr["tools"]); qv = P.vec.transform([tr["query"]]); csim = (qv @ P.capvec.T).toarray().ravel()
        R = {P.capnames[j] for j in np.argsort(-csim)[:max(1, len({c for i in truth_set for c in T[i]["caps"]}) or 1)] if csim[j] > 0}
        task = dict(type="general", domain="programming", concepts=[], level=1, R=R, D=set(), asm="none", query=tr["query"])
        prm = {**DEFAULTS, "use_flow": False, "k_max": max(4, len(truth_set))}
        kits = {"EmerKit": compose_task(P, task, labels, deny, prm, explain=False)["K"] or [],
                "EmerKit-no-interactions": compose_task(P, task, labels, deny, {**prm, "lam_I": 0.0, "lam_C": 0.0}, explain=False)["K"] or [],
                "RAG-MCP": rag_mcp(P, task, prm["k_max"]), "Zeng": zeng(P, task, prm) or []}
        for m, K in kits.items():
            inter = len(truth_set & set(K))
            rows.append(dict(seed=seed, trace=k, method=m, size=len(K), precision=inter / max(1, len(K)), recall=inter / len(truth_set),
                             jaccard=inter / max(1, len(truth_set | set(K))), load=sum(T[i]["cost"] for i in K), caps_source="llm" if keep else "mapper"))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"A7_{MODEL}_seed{seed}.csv"), index=False)


def toucan_caps():
    """LLM tagging of every Toucan tool with categories of the Toucan vocabulary (cache for A7)."""
    from emerkit.sources import toucan_catalogue
    T, S = toucan_catalogue(0); client = llm.LLMClient()
    cats = sorted({t["category"] for t in T if t.get("category")}); keys = {c: c.lower().replace(" & ", "_").replace(" ", "_").replace("/", "_") for c in cats}
    out = {}
    for t in T:
        r = llm.parse_json(client("Return only JSON.", f"Tool {t['name']}: {t['desc']}\nWhich of these categories does it serve (1 to 3)? {cats}. Answer with a JSON list.")) or []
        out[f"{t['server']}::{t['name']}"] = [keys[c] for c in r if c in keys]
    path = os.path.join(ROOT, "data", "traces", f"toucan_caps_{MODEL}.json"); os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w"))


def c1(seed):
    """C1 with language models: Compositio Prompto (adapted) and EmerKit behind the LLM workflow generator, on the requests of C2."""
    from emerkit.e2e import run_workflow, workflow_measures, validate, sub_catalogue, repair_workflow
    T, S, P = prepared(seed); client = llm.LLMClient(); deny = norm_base(); truth = P.labels_for(None, "truth")
    lab = P.labels_for(P.conformal_sets(0.05), "worst")
    def one(job):
        qi, rq = job; rows = []
        cands = sorted({i for i in range(len(T)) if T[i]["caps"] and rq["domain"] in T[i]["domains"]})
        comp = llm.compositio_prompto(client, rq, T, cands)
        if comp is None:
            rows.append(dict(seed=seed, request=qi, method="Compositio Prompto", parsed=0))
        else:
            kits = llm.resolve_names(comp, T); risky = []; ok_tasks = 0
            for K, t in zip(kits, comp["tasks"]):
                ty = t.get("type") if t.get("type") in TYPES else "practice"
                if K:
                    risky.append(int(bool(risk(T, K, truth, dict(type=ty, D=DEFAULT_D[ty]), deny))))
                    need = {r for r in t.get("capabilities", []) if r in CAPS}
                    ok_tasks += int(bool(need) and need <= {c for i in K for c in (T[i]["true_caps"] or set())})
            covered = {c for K, t in zip(kits, comp["tasks"]) if K for c in t.get("concepts", [])}
            n = max(1, len(comp["tasks"]))
            rows.append(dict(seed=seed, request=qi, method="Compositio Prompto", parsed=1, n_tasks=len(comp["tasks"]),
                             equipped=ok_tasks / n, any_dfr=int(any(risky)), obj_cov=len(set(rq["targets"]) & covered) / len(rq["targets"])))
        wf = None; raw_valid = 0
        for attempt in range(3):                                 # EmerKit behind the LLM generator (repair, validator, two regenerations)
            wf = llm.llm_generate_workflow(client, rq)
            raw_valid = int(bool(wf["tasks"]) and not validate(dict(wf, tasks=[dict(t) for t in wf["tasks"]])))
            wf = repair_workflow(wf) if wf["tasks"] else wf
            if wf["tasks"] and not validate(wf):
                break
        if not wf["tasks"] or validate(wf):
            rows.append(dict(seed=seed, request=qi, method="LLM-gen+EmerKit", parsed=0)); return rows
        b = run_workflow(P, wf, lab, deny, DEFAULTS, np.random.default_rng(qi))
        eqt = [p for p in b["plan"].values() if p["K"]]
        rows.append(dict(seed=seed, request=qi, method="LLM-gen+EmerKit", parsed=1, n_tasks=len(wf["tasks"]), raw_valid=raw_valid, repairs=wf.get("repairs", 0),
                         equipped=len(eqt) / len(wf["tasks"]), any_dfr=int(any(risk(T, p["K"], truth, p["task"], deny) for p in eqt)),
                         **workflow_measures(b, rq["targets"], sub_catalogue("medium"))))
        return rows
    out = [r for rows in llm.pmap(one, list(enumerate(generate_requests(seed, n_per_domain=8)))) for r in rows]
    pd.DataFrame(out).to_csv(os.path.join(OUT, f"C1_{MODEL}_seed{seed}.csv"), index=False)


JUDGE_SYSTEM = ("You are an experienced university instructor and learning designer. Answer only with JSON. Base your judgement "
                "on established pedagogy (Bloom's revised taxonomy, constructive alignment, the assistance dilemma) and on data "
                "protection rules for learners (GDPR, FERPA).")


def judge_value(txt, key):
    """Value of `key` in a JSON answer; robust to answers cut by the token limit."""
    m = re.search(r'"%s"\s*:\s*(true|false|[0-9.]+)' % key, txt or "")
    if not m: return None
    v = m.group(1)
    return v == "true" if v in ("true", "false") else float(v)


def judges():
    """Each model rates every cell of the fit matrix, every substitution pair and every norm cell; agreement is computed across models."""
    client = llm.LLMClient(); jobs = []
    for act, by in PHI.items():
        for ty, v in by.items():
            jobs.append(("fit", act, ty, v, f"How well does a tool whose pedagogical act is '{act}' fit a '{ty}' learning task? "
                                               f"Return {{\"fit\": number between 0 and 1, \"rationale\": short text}}.", "fit"))
    for a_ in TYPES:
        for b_ in TYPES:
            jobs.append(("sub", a_, b_, b_ in SUB.get(a_, []), f"Can a '{a_}' task be replaced by a '{b_}' task on the same concepts without "
                                                   f"changing its learning objective? Return {{\"admissible\": true/false, \"rationale\": ...}}.", "admissible"))
    for (ty, att, role), d in norm_base().items():
        jobs.append(("norm", f"{ty}|{att}", role, not d, f"In a '{ty}' task, may learner data of kind '{att}' (ID=identity, PROF=profile, "
                     f"PERF=performance, BEH=behaviour, CONT=content produced) flow to a receiver of role '{role}' (INS=institution, "
                     f"PRO=contracted processor, EXT=third party, PUB=public web)? Return {{\"allowed\": true/false, \"rationale\": ...}}.", "allowed"))
    def run(j):
        kind, a_, b_, cur, q, key = j
        txt = client(JUDGE_SYSTEM, q + " Keep the rationale to one sentence of at most 25 words.", max_tokens=500)
        return dict(kind=kind, a=a_, b=b_, model=MODEL, value=judge_value(txt, key), current=cur, rationale=txt[:600])
    pd.DataFrame(llm.pmap(run, jobs)).to_csv(os.path.join(OUT, f"judges_{MODEL}.csv"), index=False)


def audit_review():
    """LLM review of the evidence gathered by audit_static.py for each sampled tool -> data/audit/labels_llm_<model>.jsonl."""
    client = llm.LLMClient(); jobs = []
    for line in open(os.path.join(ROOT, "data", "audit", "static_evidence.jsonl")):
        rec = json.loads(line)
        if not rec.get("fetched"): continue
        for name, ev in rec["tools"].items():
            jobs.append((rec, name, ev))
    def run(j):
        rec, name, ev = j
        q = (f"MCP server {rec['server']} ({rec['repository']}); tool '{name}'. Static evidence: {json.dumps(ev)}; server-level outbound "
             f"calls={rec['net']}, external hosts={rec['urls'][:10]}, storage={rec['store']}. For each label answer 0, 1 or \"unknown\": "
             f"out (sends what it receives beyond its own server), src_ID, src_PROF, src_PERF, src_BEH, src_CONT (returns that kind of "
             f"personal data of the user). Return only a JSON object {{label: value}}.")
        return dict(server=rec["server"], tool=name, model=MODEL, labels=llm.parse_json(client(JUDGE_SYSTEM, q, max_tokens=1500)))
    with open(os.path.join(ROOT, "data", "audit", f"labels_llm_{MODEL}.jsonl"), "w") as f:
        for r in llm.pmap(run, jobs):
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    fn = {"probes-design": probes_design, "requests-bank": requests_bank, "probes": probes, "fit-probes": fit_probes, "a6": a6, "capabilities": capabilities,
          "toucan": toucan, "toucan-caps": toucan_caps, "a7": a7, "c1": c1, "judges": judges, "audit-review": audit_review}[cmd]
    fn(*[int(a) if a.isdigit() else a for a in args])
