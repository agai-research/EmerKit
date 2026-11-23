"""Series F1: live enactment through the MCP gateway.
policy=exhaustive (no model): every tool is called with every piece of context its parameters can carry, so every flow that the
toolkit makes possible is actually realised; observed flows (canaries found in the sinks) must equal predicted flows.
policy=llm: an LLM agent enacts the task through the gateway (run in Claude Code).
Usage: python scripts/run_f1.py SEED N_TASKS [exhaustive|llm]"""
import sys, os, json, asyncio, tempfile, shutil, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from emerkit.data import synth_catalogue, sample_tasks, PARAM_FOR, FREE_PARAMS
from emerkit.prep import Prepared
from emerkit.vocab import norm_base, ATT, EXT_ROLES
from emerkit.compose import compose_task, DEFAULTS, src
from emerkit.runtime.servers import canary, ATTR_TEXT

PARAM_ATTR = {p: a for a, ps in PARAM_FOR.items() for p in ps}


def predicted_flows(T, K, truth, task):
    A0 = set(task["D"]) | set(task.get("delta_in", set()))
    A = A0 | set().union(*[src(truth[i]) for i in K] or [set()])
    flows = set()
    for i in K:
        others = A0 | set().union(*[src(truth[j]) for j in K if T[j]["server"] != T[i]["server"]] or [set()])
        flows |= {(a, T[i]["role"]) for a in others & T[i]["acc"]}
        if truth[i]["out"]:
            flows |= {(a, r) for a in A & T[i]["acc"] for r in EXT_ROLES}
    return flows


def arg_for(param, context):
    """What a parameter can carry: a free-text field carries everything; a typed field carries its attribute class only."""
    if param in FREE_PARAMS:
        return " ".join(context)
    a = PARAM_ATTR.get(param)
    return " ".join(x for x in context if a and f"CANARY-{a}-" in x)


async def enact(cat_path, sink, log, K_ids, T_by_id, task, policy):
    params = StdioServerParameters(command=sys.executable, args=["-m", "emerkit.runtime.gateway", cat_path, sink, log] + K_ids,
                                   env={**os.environ, "PYTHONPATH": ROOT}, cwd=ROOT)
    context = [ATTR_TEXT[a].format(c=canary(a, "ctx")) for a in sorted(set(task["D"]) | set(task.get("delta_in", set())))]
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            listed = {t.name for t in (await s.list_tools()).tools}
            expected = {f"{T_by_id[i]['name']}__{i}" for i in K_ids}
            try:
                res = await s.call_tool("not_in_toolkit__x", {})
                blocked = int(bool(getattr(res, "isError", False)))
            except Exception:
                blocked = 1
            if policy == "exhaustive":
                for _ in range(2):                      # sources first, then sinks see everything the sources returned
                    for i in K_ids:
                        t = T_by_id[i]
                        res = await s.call_tool(f"{t['name']}__{i}", {p: arg_for(p, context) for p in t["params"]})
                        txt = "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
                        context += [line.replace("CANARY-", "CANARY-").replace(")", f" src={t['server']})") for line in txt.splitlines()]
            else:
                from emerkit.llm_agent import run_agent
                await run_agent(s, task, context)
    return listed == expected, blocked


def observed_flows(sink, T_by_id):
    flows = set()
    for role in ["INS", "PRO", "EXT", "PUB"]:
        f = os.path.join(sink, f"sink_{role}.jsonl")
        if not os.path.exists(f): continue
        for line in open(f):
            rec = json.loads(line); srv = rec["server"]
            for a in ATT:
                for piece in rec["data"].split("("):
                    if f"CANARY-{a}-" in piece:
                        own = f"src={srv})" in piece or f"src={srv} " in piece
                        if rec.get("forwarded") or not own:        # own-server data is not a flow to the own role
                            flows.add((a, role))
    return flows


def main(seed, n, policy="exhaustive"):
    T, S = synth_catalogue(seed); P = Prepared(T, S, seed=seed); deny = norm_base()
    truth = P.labels_for(None, "truth"); lab = P.labels_for(P.conformal_sets(0.05), "worst")
    cat_path = os.path.join(ROOT, "data", "generated", f"catalogue_synthetic_seed{seed}.json")
    T_by_id = {t["id"]: t for t in T}
    rows = []
    tasks = sample_tasks(seed, 300); step = max(1, len(tasks) // n)
    for k, task in list(enumerate(tasks))[::step][:n]:              # systematic sample across pedagogical types
        r = compose_task(P, task, lab, deny, DEFAULTS, explain=False)
        if not r["K"]:
            continue
        K_ids = [T[i]["id"] for i in r["K"]]
        tmp = tempfile.mkdtemp(); sink = os.path.join(tmp, "sink"); log = os.path.join(tmp, "calls.jsonl")
        t0 = time.time()
        exposed_ok, blocked = asyncio.run(enact(cat_path, sink, log, K_ids, T_by_id, task, policy))
        obs = observed_flows(sink, T_by_id); pred = predicted_flows(T, r["K"], truth, task)
        denied_obs = {f for f in obs if deny[(task["type"], f[0], f[1])]}
        rows.append(dict(seed=seed, task=k, type=task["type"], policy=policy, n_tools=len(K_ids), exposed_only_K=int(exposed_ok),
                         call_outside_blocked=blocked, predicted=len(pred), observed=len(obs), exact_match=int(obs == pred),
                         missed=len(pred - obs), unexpected=len(obs - pred), denied_observed=len(denied_obs), seconds=time.time() - t0))
        shutil.rmtree(tmp)
    out = os.path.join(ROOT, "results", "f1"); os.makedirs(out, exist_ok=True)
    tag = policy if policy == "exhaustive" else f"llm_{os.environ.get('EMERKIT_MODEL', 'model').replace('/', '_').replace(':', '-')}"
    pd.DataFrame(rows).to_csv(os.path.join(out, f"F1_{tag}_seed{seed}.csv"), index=False)
    d = pd.DataFrame(rows); print(d[["exposed_only_K", "call_outside_blocked", "exact_match", "missed", "unexpected", "denied_observed", "seconds"]].mean().round(3).to_dict(), len(d))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "exhaustive")
