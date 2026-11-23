"""Series A7: toolkits for real requests vs the tools actually called in Toucan trajectories (no model call).
Capabilities of a real tool = the category labels of its server (Toucan metadata); the requirements of a request = its 2 closest
categories. Usage evidence (eta) and co-usage come from the fit traces only. Two splits: 'warm' (prompt groups held out: tools
have usage history) and 'cold' (servers held out: no history). All methods return at most k = |tools actually called| tools.
Usage: python scripts/run_a7.py [warm|cold] [MAX_TEST]"""
import os, sys, json, re
import numpy as np, pandas as pd
from collections import defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from emerkit.sources import toucan_catalogue
from emerkit.prep import Prepared
from emerkit.compose import compose_task, rag_mcp, zeng, DEFAULTS

def slug(x): return re.sub(r"[^a-z0-9]+", "_", x.lower()).strip("_")

def main(split="warm", max_test=600):
    T, S = toucan_catalogue(0)
    key = {f"{t['server'][7:]}::{t['name']}": i for i, t in enumerate(T)}
    vocab = {slug(l): l for t in T for l in t.get("labels", [])}
    for t in T:
        t["caps"] = {slug(l) for l in t.get("labels", [])}
    traces = [json.loads(l) for l in open(os.path.join(ROOT, "data", "traces", "toucan_traces.jsonl"))]
    traces = [dict(x, idx=[key[k] for k in x["tools"] if k in key]) for x in traces if x["success"]]
    traces = [x for x in traces if x["idx"]]
    rng = np.random.default_rng(0)
    if split == "warm":
        groups = sorted({x["group"] for x in traces}); rng.shuffle(groups); test_g = set(groups[: int(0.3 * len(groups))])
        fit = [x for x in traces if x["group"] not in test_g]; test = [x for x in traces if x["group"] in test_g]
    else:
        servers = sorted({T[i]["server"] for x in traces for i in x["idx"]}); rng.shuffle(servers); test_s = set(servers[: int(0.3 * len(servers))])
        fit = [x for x in traces if all(T[i]["server"] not in test_s for i in x["idx"])]
        test = [x for x in traces if all(T[i]["server"] in test_s for i in x["idx"])]
    P = Prepared(T, S, seed=0, cap_vocab=vocab, keep_caps=True)
    Pn = Prepared(T, S, seed=0, cap_vocab=vocab, keep_caps=True)            # same catalogue, no traces
    P.set_traces([dict(query=x["query"], tools=x["idx"], success=True) for x in fit])
    deny = defaultdict(bool); labels = [{l: 0 for l in ["src_ID", "src_PROF", "src_PERF", "src_BEH", "src_CONT", "out"]}] * len(T)
    rows = []
    for n, x in enumerate(test[: int(max_test)]):
        truth = set(x["idx"]); k = len(truth)
        csim = (P.vec.transform([x["query"]]) @ P.capvec.T).toarray().ravel()
        R = {P.capnames[j] for j in np.argsort(-csim)[:2] if csim[j] > 0}
        task = dict(type="general", domain="programming", concepts=[], level=1, R=R, D=set(), asm="none", query=x["query"])
        prm = {**DEFAULTS, "use_flow": False, "use_qos": False, "k_max": max(k, len(R)), "N": 30}
        kits = {"EmerKit (traces)": compose_task(P, task, labels, deny, {**prm, "eta": 1.0}, explain=False)["K"],
                "EmerKit (traces, no co-usage)": None,
                "EmerKit (no traces)": compose_task(Pn, task, labels, deny, {**prm, "eta": 1.0}, explain=False)["K"],
                "RAG-MCP": rag_mcp(P, task, k), "Zeng": zeng(P, task, prm)}
        co, P.co_count = P.co_count, {}                                       # ablation: evidence kept, co-usage removed
        kits["EmerKit (traces, no co-usage)"] = compose_task(P, task, labels, deny, {**prm, "eta": 1.0}, explain=False)["K"]
        P.co_count = co
        for m, K in kits.items():
            K = list(K or [])[: max(k, len(R))]; inter = len(truth & set(K))
            rows.append(dict(split=split, trace=n, method=m, k=k, size=len(K), hit=int(inter > 0), precision=inter / max(1, len(K)),
                             recall=inter / k, jaccard=inter / max(1, len(truth | set(K)))))
    d = pd.DataFrame(rows); out = os.path.join(ROOT, "results", "llm"); os.makedirs(out, exist_ok=True)
    d.to_csv(os.path.join(out, f"A7_{split}.csv"), index=False)
    print(split, "fit traces", len(fit), "test traces", min(len(test), int(max_test)))
    print(d.groupby("method")[["size", "hit", "precision", "recall", "jaccard"]].mean().round(3).to_string())

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "warm", sys.argv[2] if len(sys.argv) > 2 else 600)
