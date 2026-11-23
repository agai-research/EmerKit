"""Series A6 on real tools (Toucan catalogue): does interference-aware composition help an agent choose the right tool?
For each of the 500 targeted requests of A5, the 16 tools most relevant to the request form a common candidate pool. Three toolkits
of k = 8 tools are built from it: RAG-MCP keeps the 8 most relevant; EmerKit maximizes relevance minus the fitted interference
(lambda_I * sum of iota over pairs, Sect. 4.4), exactly by enumeration of the 12,870 subsets; Random draws 8 tools. The agent
(model in $EMERKIT_MODEL) then chooses one tool. Success = the target or one of its clones (same name, description similarity >= 0.95).
Usage: python scripts/run_a6_real.py build | run METHOD [MAX] | summary"""
import os, sys, json, itertools, glob
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
from emerkit.sources import toucan_catalogue
from emerkit.prep import Prepared
from emerkit import llm
POOL, K = 16, 8
DIR = os.path.join(ROOT, "data", "probes_real"); OUT = os.path.join(ROOT, "results", "llm"); MODEL = os.environ.get("EMERKIT_MODEL", "unset")


def build():
    T, S = toucan_catalogue(0); P = Prepared(T, S, seed=0)
    fw = json.load(open(os.path.join(ROOT, "config", "fitted_weights.json")))
    lam, beta, s0 = fw["lam_I"], fw["beta"], fw["omega0"]
    reqs = [json.loads(l) for l in open(os.path.join(DIR, "requests.jsonl"))]
    rng = np.random.default_rng(6); rows = []
    for k, r in enumerate(reqs):
        rel = (P.X @ P.vec.transform([r["request"]]).T).toarray().ravel()
        pool = [int(i) for i in np.argsort(-rel)[:POOL]]
        sim = np.array([[P.desc_sim(a, b) for b in pool] for a in pool])
        iota = 1 / (1 + np.exp(-beta * (sim - s0))); np.fill_diagonal(iota, 0.0)
        relp = np.array([rel[i] for i in pool]); best, bestv = list(range(K)), -1e18
        combos = itertools.combinations(range(POOL), K) if iota.max() > 1e-6 else []   # no interference: relevance order is optimal
        for c in combos:                     # exact maximization of the composer objective
            idx = list(c); v = relp[idx].sum() - lam * iota[np.ix_(idx, idx)].sum() / 2
            if v > bestv: best, bestv = idx, v
        clones = [i for i in pool if T[i]["name"] == T[r["target"]]["name"] and P.desc_sim(i, r["target"]) >= 0.95]
        rows.append(dict(probe=k, target=r["target"], request=r["request"], pool=pool, clones=sorted(set(clones) | {r["target"]}),
                         kits={"RAG-MCP": pool[:K], "EmerKit": [pool[j] for j in best], "Random": [int(x) for x in rng.choice(pool, K, replace=False)]}))
    json.dump(rows, open(os.path.join(DIR, "a6_toolkits.json"), "w"))
    print(len(rows), "requests; target in pool:", np.mean([r["target"] in r["pool"] for r in rows]).round(3),
          {m: np.mean([bool(set(r["kits"][m]) & set(r["clones"])) for r in rows]).round(3) for m in ["RAG-MCP", "EmerKit", "Random"]})


def run(method, maxn=None):
    T, S = toucan_catalogue(0); client = llm.LLMClient()
    rows = json.load(open(os.path.join(DIR, "a6_toolkits.json")))[: int(maxn) if maxn else None]
    def one(r):
        kit = r["kits"][method]; c = llm.probe(client, r["request"], kit, T)
        avail = bool(set(kit) & set(r["clones"]))
        return dict(probe=r["probe"], method=method, model=MODEL, available=int(avail), chosen=c, success=int(c in r["clones"]),
                    clone_pairs=sum(1 for a, b in itertools.combinations(kit, 2) if T[a]["name"] == T[b]["name"]))
    pd.DataFrame(llm.pmap(one, rows)).to_csv(os.path.join(OUT, f"A6real_{method}_{MODEL}.csv"), index=False)


def summary():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "A6real_*.csv"))])
    g = d.groupby("method").agg(success=("success", "mean"), available=("available", "mean"), clone_pairs=("clone_pairs", "mean"))
    g["success_when_available"] = d[d.available == 1].groupby("method").success.mean()
    print(g.round(3).to_string())


if __name__ == "__main__":
    a = sys.argv[1:]
    {"build": build, "run": lambda: run(a[1], a[2] if len(a) > 2 else None), "summary": summary}[a[0]]()
