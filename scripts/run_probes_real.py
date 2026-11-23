"""Series A5 on real tool definitions (Toucan catalogue). For each target tool, a model that is not probed writes a request that
only the target can serve; the probed model sees the target among k-1 other real tools, either the most similar tools of other
servers (confusable) or random tools. Correct = the probed model picks the target.
Usage: python scripts/run_probes_real.py requests N_TARGETS     (writer model in $EMERKIT_MODEL)
       python scripts/run_probes_real.py probe                  (probed model in $EMERKIT_MODEL)
       python scripts/run_probes_real.py fit"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from emerkit.sources import toucan_catalogue
from emerkit.prep import Prepared
from emerkit import llm
DIR = os.path.join(ROOT, "data", "probes_real"); os.makedirs(DIR, exist_ok=True)
OUT = os.path.join(ROOT, "results", "llm"); MODEL = os.environ.get("EMERKIT_MODEL", "unset")
KS, B, THETA = (4, 8, 16, 32, 64), 2000, 0.75


def catalogue():
    T, S = toucan_catalogue(0)
    keep = [i for i, t in enumerate(T) if len(t["desc"]) >= 40]           # tools with a usable description
    return T, S, keep


def requests(n):
    T, S, keep = catalogue(); rng = np.random.default_rng(0); client = llm.LLMClient()
    targets = [int(i) for i in rng.choice(keep, int(n), replace=False)]
    def run(i):
        t = T[i]
        q = (f"Tool '{t['name']}' of the MCP server '{t['server'][7:]}': {t['desc'][:600]}\nWrite one realistic request (one or two sentences) "
             f"that a user could send to an AI assistant and that this tool, and not a generic tool, should serve: mention the concrete "
             f"service, data source or object it works on. Do not copy the tool name. Return only the request.")
        return dict(target=i, tool=t["name"], server=t["server"], request=client("Return only the request text.", q, max_tokens=300).strip())
    rows = llm.pmap(run, targets)
    with open(os.path.join(DIR, "requests.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(len(rows), "requests written by", MODEL)


def design():
    T, S, keep = catalogue(); P = Prepared(T, S, seed=0); rng = np.random.default_rng(1)
    reqs = [json.loads(l) for l in open(os.path.join(DIR, "requests.jsonl"))]
    rows = []
    for k_, r in enumerate(reqs):
        i = r["target"]; k = KS[k_ % len(KS)]; conf = (k_ // len(KS)) % 2
        others = [j for j in keep if T[j]["server"] != T[i]["server"]]
        if conf:
            sims = (P.D[others] @ P.D[i].T).toarray().ravel(); dist = [others[j] for j in np.argsort(-sims)[:k - 1]]
        else:
            dist = [int(j) for j in rng.choice(others, k - 1, replace=False)]
        kit = dist + [i]; rng.shuffle(kit)
        load = sum(T[j]["cost"] for j in kit) + sum(S[s]["preamble"] for s in {T[j]["server"] for j in kit})
        inter = sum(P.desc_sim(a, b) for x, a in enumerate(kit) for b in kit[x + 1:] if a == i or b == i)   # similarity to the target
        rows.append(dict(probe=k_, target=i, request=r["request"], k=k, confusable=conf, toolkit=[int(j) for j in kit], load=load,
                         target_similarity=inter, saturation=max(0.0, (load - THETA * B) / ((1 - THETA) * B)) ** 2, split="fit" if k_ % 2 == 0 else "test"))
    return T, rows


def probe():
    T, rows = design(); client = llm.LLMClient()
    def run(x):
        c = llm.probe(client, x["request"], x["toolkit"], T)
        return dict({k: v for k, v in x.items() if k != "toolkit"}, model=MODEL, chosen=c, correct=int(c == x["target"]))
    pd.DataFrame(llm.pmap(run, rows)).to_csv(os.path.join(OUT, f"A5real_{MODEL}.csv"), index=False)


def fit():
    import glob, statsmodels.formula.api as smf
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "A5real_*.csv"))])
    d["log_load"] = np.log(d.load)
    print(d.groupby(["model", "confusable"]).correct.mean().round(3).unstack().to_string())
    print(d.groupby(["model", "k"]).correct.mean().round(3).unstack().to_string())
    for f in ["correct ~ confusable + log_load + C(model)", "correct ~ target_similarity + saturation + C(model)"]:
        r = smf.logit(f, d[d.split == "fit"]).fit(disp=0)
        print(f, "\n", pd.DataFrame({"coef": r.params, "se": r.bse, "p": r.pvalues}).round(4).to_string())
    json.dump({"note": "logit on the fit half; see A5real_*.csv"}, open(os.path.join(ROOT, "config", "fitted_weights_real.json"), "w"))


if __name__ == "__main__":
    {"requests": lambda: requests(sys.argv[2]), "probe": probe, "fit": fit}[sys.argv[1]]()
