"""Builds portal/bundle.json, the registry and the prepared artefacts that the browser engine of the portal uses.
The preparation phase of EmerKit (capability extraction, weak supervision, conformal prediction sets, description similarities)
runs here, offline, exactly as in the experiments; the request phase (decomposition, workflow, composition, flow check, revision)
runs live in the browser (portal/engine.js). Two registries are prepared:
  * emerkit  : the synthetic educational registry of the experiments (seed 0), with its ground truth;
  * real     : the same registry plus the 6,044 real tools of the Toucan MCP server catalogue (mixed preparation of series E1).
Usage: python portal/build_bundle.py"""
import os, sys, json, math
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from emerkit.data import synth_catalogue, LABELS
from emerkit.sources import toucan_catalogue
from emerkit.prep import Prepared
from emerkit import vocab as V
from emerkit.compose import DEFAULTS

SEED = 0
LEVELS = [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]
EH = json.load(open(os.path.join(ROOT, "results", f"ehat_seed{SEED}.json")))


def masks(G, i):
    """For each level: bit k set in 'pos' when label k may be 1 (worst case), in 'unc' when the set is {0,1}."""
    out = []
    for e, g in G:
        pos = unc = 0
        for k, l in enumerate(LABELS):
            s = g[l][i]
            if 1 in s: pos |= 1 << k
            if len(s) > 1: unc |= 1 << k
        out.append([pos, unc])
    return out


def registry(P, servers, with_truth):
    T = P.tools
    Gs = [(e, P.conformal_sets(e)) for e in LEVELS]
    capidx = {c: j for j, c in enumerate(P.capnames)}
    active = [i for i, t in enumerate(T) if t["caps"]]
    tools = []
    for i, t in enumerate(T):
        rec = dict(id=t["id"], name=t["name"], server=t["server"], role=t["role"], desc=(t["desc"] or "")[:260], caps=sorted(t["caps"]),
                   act=t["act"], cost=int(t["cost"]), lat=float(t["lat"]), price=float(t["price"]), rel=float(t["rel"]),
                   acc=sorted(t["acc"]), prod=sorted(t["prod"]), cons=sorted(t["cons"]), domains=sorted(t["domains"]), real=not t["synthetic"],
                   declared=[t["hints"][l] for l in LABELS], m=masks(Gs, i), audited=i in P.cal_set)
        if i in P.cal_set:                                          # audited calibration tools: labels known (as in Prepared.labels_for)
            pos = sum(1 << k for k, l in enumerate(LABELS) if t["truth"][l]); rec["m"] = [[pos, 0] for _ in LEVELS]
        if t["caps"]:
            rec["capsim"] = {c: float(P.capsim[i, capidx[c]]) for c in t["caps"]}
        if with_truth and t["truth"] is not None:
            rec["truth"] = [int(t["truth"][l]) for l in LABELS]
        if not t["synthetic"]:
            rec["category"] = t.get("category"); rec.pop("acc"); rec["acc"] = sorted(t["acc"])
        tools.append(rec)
    sims = []                                                       # pairs whose interference can exceed 1e-3 (similarity > 0.40)
    for a in range(len(active)):
        for b in range(a + 1, len(active)):
            i, j = active[a], active[b]
            if T[i]["caps"] & T[j]["caps"]:
                s = P.desc_sim(i, j)
                if s > 0.40: sims.append([i, j, s])
    srv = {k: dict(role=v["role"], avail=float(v["avail"]), preamble=int(v["preamble"]), real=not v.get("synthetic", True),
                   category=v.get("category"), repository=v.get("repository")) for k, v in servers.items()}
    return dict(tools=tools, servers=srv, sims=sims, n_active=len(active))


def main():
    T, S = synth_catalogue(SEED); P = Prepared(T, S, seed=SEED)
    reg_e = registry(P, S, True)
    R, RS = toucan_catalogue(SEED); Tm, Sm = synth_catalogue(SEED)
    Pm = Prepared(Tm + R, {**Sm, **RS}, seed=SEED)
    reg_r = registry(Pm, {**Sm, **RS}, True)
    caps = {c: dict(text=v[0], domains=sorted(v[1]) if v[1] else None, src=sorted(v[2] or []), acc=sorted(v[3] or []), prod=sorted(v[4] or []), cons=sorted(v[5] or []), act=v[6])
            for c, v in V.CAPS.items()}
    norms = {p: {f"{ty}|{a}|{r}": bool(d) for (ty, a, r), d in V.norm_base(p).items()} for p in ["permissive", "standard", "strict"]}
    import pandas as pd
    tr = pd.read_csv(os.path.join(ROOT, "results", f"b1_trace_seed{SEED}.csv"))
    stats = {str(float(e)): dict(risk=float(g.test_dfr.mean()), equipped=float(g.test_eq.mean())) for e, g in tr.groupby("eps") if float(e) in LEVELS}
    bundle = dict(
        meta=dict(seed=SEED, levels=LEVELS, eps_hat={"0.05": EH["e_hat"], "0.10": EH.get("e_hat10", EH["e_hat"])}, labels=LABELS, level_stats=stats,
                  defaults={k: v for k, v in DEFAULTS.items() if isinstance(v, (int, float, bool))}),
        vocab=dict(types=V.TYPES, domains=V.DOMAINS, attributes=V.ATT, roles=V.ROLES, ext_roles=sorted(V.EXT_ROLES), caps=caps,
                   type_caps=V.TYPE_CAPS, default_d={k: sorted(v) for k, v in V.DEFAULT_D.items()}, level=V.LEVEL, asm=V.ASM,
                   core_cap={f"{k[0]}|{k[1]}": v for k, v in V.CORE_CAP.items()}, handoff={k: sorted(v) for k, v in V.HANDOFF_ATTRS.items()},
                   sub=V.SUB, phi=V.PHI, concepts={d: [[c, p] for c, p in V.DOMAIN_CONCEPTS[d]] for d in V.DOMAINS}),
        norms=norms, registries=dict(emerkit=reg_e, real=reg_r))
    path = os.path.join(ROOT, "portal", "bundle.json")
    json.dump(bundle, open(path, "w"), separators=(",", ":"))
    print("bundle", os.path.getsize(path) // 1024, "KB;", "emerkit tools", len(reg_e["tools"]), "active", reg_e["n_active"], "sims", len(reg_e["sims"]),
          "| real registry tools", len(reg_r["tools"]), "active", reg_r["n_active"], "sims", len(reg_r["sims"]))


if __name__ == "__main__":
    main()
