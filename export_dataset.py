"""Exports the datasets used by the experiments to data/generated/ (JSON): catalogues, tasks, requests, workflows, archetypes, norms."""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emerkit.data import synth_catalogue, sample_tasks, generate_requests, archetypes
from emerkit.vocab import norm_base, CAPS, TYPE_CAPS, SUB, PHI, DOMAIN_CONCEPTS, ATT, ROLES, TYPES
from emerkit.e2e import generate_candidates
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "generated"); os.makedirs(OUT, exist_ok=True)
def ser(o):
    if isinstance(o, set): return sorted(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.str_): return str(o)
    raise TypeError(type(o))
def dump(name, obj): json.dump(obj, open(os.path.join(OUT, name), "w"), default=ser, ensure_ascii=False, indent=1)
dump("vocabulary.json", dict(attributes=ATT, roles=ROLES, types=TYPES, capabilities={k: dict(description=v[0], domains=v[1], src=v[2], acc=v[3], produces=v[4], consumes=v[5], act=v[6]) for k, v in CAPS.items()},
                            type_capabilities=TYPE_CAPS, substitution_catalogue=SUB, fit_matrix=PHI, concepts=DOMAIN_CONCEPTS))
for p in ["permissive", "standard", "strict"]:
    dump(f"norms_{p}.json", [dict(type=t, attribute=a, role=r, deny=d) for (t, a, r), d in norm_base(p).items()])
for seed in range(5):
    T, S = synth_catalogue(seed)
    dump(f"catalogue_synthetic_seed{seed}.json", dict(servers=S, tools=T))
    dump(f"tasks_test_seed{seed}.json", sample_tasks(seed, 300)); dump(f"tasks_calibration_seed{seed}.json", sample_tasks(seed + 5000, 204)[:200])
    dump(f"archetypes_seed{seed}.json", archetypes(seed))
    rq = generate_requests(seed, n_per_domain=8); dump(f"requests_seed{seed}.json", rq)
    dump(f"workflows_seed{seed}.json", [generate_candidates(np.random.default_rng(seed * 1000 + i), q, 3) for i, q in enumerate(rq)])
print(len(os.listdir(OUT)), "files")
