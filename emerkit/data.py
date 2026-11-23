"""Data generation for the EmerKit experiments.

Synthetic educational MCP servers carry ground truth (capabilities, data labels, roles, confusable pairs);
real tools from MCP-tools carry none of it and enter as uncertain, third-party candidates.
"""
import json, re, math
import numpy as np
from .vocab import (ATT, ROLES, TYPES, ASM, LEVEL, DEFAULT_D, DOMAINS, CAPS, TYPE_CAPS, CORE_CAP,
                    HANDOFF_ATTRS, DOMAIN_CONCEPTS)

LABELS = [f"src_{a}" for a in ATT] + ["out"]
VENDORS = ["nova", "atlas", "lumen", "quanta", "orbit", "cedar", "pixel", "vertex", "harbor", "zenith", "maple", "delta",
           "kite", "ember", "sable", "tidal", "aurora", "cobalt", "fable", "granite"]
TEMPLATES = ["{kw}.", "Use this tool to {kw}.", "Tool that can {kw} for learners.", "Lets the assistant {kw}.",
             "Service to {kw} quickly.", "Helps students: {kw}.", "{kw} with configurable options."]
OUT_CUES = ["Results are synced to the vendor cloud.", "Shares content with partner services.",
            "Sends the input to an external API for processing.", "Uploads data to our servers."]
SRC_CUES = {"PERF": "Returns the score obtained.", "ID": "Returns the learner identifier.", "BEH": "Includes the activity history.",
            "PROF": "Returns profile fields.", "CONT": "Returns the full output text."}
PARAM_FOR = {"ID": ["student_id", "learner_id", "user_id"], "PERF": ["grade", "score"], "CONT": ["text", "content", "answer", "code"],
             "PROF": ["preferences", "level"], "BEH": ["activity_log"]}
FREE_PARAMS = ["query", "payload", "prompt", "message"]
OUT_PARAMS = ["callback_url", "recipient_email", "webhook"]
FILLER = (" Inputs are validated against the schema and errors are returned with a code."
          " The tool follows the Model Context Protocol specification and supports pagination where relevant.")
PAIRS = [("generate_quiz", "grade_answers"), ("run_code", "grade_code"), ("record_grade", "read_gradebook"),
         ("submit_assignment", "plagiarism_check"), ("check_grammar", "writing_feedback"), ("plot_data", "load_dataset")]
INS_HEAVY = {"record_grade", "read_gradebook", "read_profile", "submit_assignment", "forum_post", "learning_analytics",
             "retrieve_courseware", "peer_review"}
PUB_OK = {"web_search", "video_lookup", "dictionary_lookup", "translate_text"}


SYN = {"explain": ["clarify", "walk through"], "concept": ["notion", "idea"], "examples": ["samples", "cases"],
       "retrieve": ["fetch", "get"], "course": ["class", "module"], "lesson": ["unit", "chapter"], "search": ["look up", "query"],
       "find": ["discover", "locate"], "videos": ["clips", "recordings"], "generate": ["create", "produce"], "quiz": ["test", "assessment"],
       "questions": ["items", "prompts"], "grade": ["evaluate", "mark"], "answers": ["responses", "replies"], "score": ["result", "mark"],
       "learner": ["student", "user"], "gradebook": ["grade register", "records"], "read": ["load", "access"], "profile": ["account details", "settings"],
       "hint": ["clue", "nudge"], "solution": ["answer key", "result"], "flashcards": ["memory cards", "review cards"], "post": ["publish", "write"],
       "forum": ["board", "discussion space"], "reviews": ["evaluations", "critiques"], "submission": ["work", "upload"], "plagiarism": ["copying", "originality"],
       "submit": ["hand in", "deliver"], "assignment": ["homework", "coursework"], "notify": ["alert", "inform"], "instructor": ["teacher", "tutor"],
       "track": ["monitor", "log"], "activity": ["usage", "actions"], "export": ["download", "output"], "report": ["summary", "statement"],
       "execute": ["run", "evaluate"], "code": ["program", "script"], "sandbox": ["isolated runtime", "container"], "tests": ["checks", "test cases"],
       "analysis": ["inspection", "review"], "queries": ["statements", "requests"], "database": ["data store", "DB"], "dataset": ["data file", "table"],
       "notebook": ["Jupyter", "worksheet"], "plot": ["draw", "chart"], "charts": ["graphs", "figures"], "grammar": ["syntax", "language rules"],
       "spelling": ["typos", "orthography"], "feedback": ["comments", "suggestions"], "essay": ["composition", "text"], "translate": ["convert", "render"],
       "definitions": ["meanings", "senses"], "pronunciation": ["speech", "accent"], "recording": ["audio", "voice sample"], "conversation": ["dialogue", "chat"],
       "speaking": ["oral", "spoken"]}


def paraphrase(text, rng, p=0.5):
    out = []
    for wd in text.split(" "):
        key = wd.lower().strip(".,")
        if key in SYN and rng.random() < p:
            out.append(SYN[key][int(rng.integers(len(SYN[key])))])
        else:
            out.append(wd)
    return " ".join(out)


def approx_tokens(obj):
    return max(1, int(round(len(json.dumps(obj, ensure_ascii=False)) / 4)))


def _qos(rng, role):
    lat = float(np.clip(np.exp(rng.normal(math.log(300), 0.8)), 50, 2000))
    price = {"INS": 0.0, "PRO": rng.uniform(0, 0.004), "EXT": rng.uniform(0, 0.01), "PUB": rng.uniform(0, 0.002)}[role]
    rel = float(rng.uniform(0.85, 0.999))
    return lat, float(price), rel


def _role_for(cap, rng, ext_bias=1.0):
    if cap in PUB_OK:
        roles, p = ["PUB", "EXT", "PRO"], [0.4, 0.4 * ext_bias, 0.2]
    elif cap in INS_HEAVY:
        roles, p = ["INS", "PRO", "EXT"], [0.6, 0.3, 0.1 * ext_bias]
    else:
        roles, p = ["INS", "PRO", "EXT"], [0.3, 0.3, 0.4 * ext_bias]
    p = np.asarray(p) / sum(p)
    return rng.choice(roles, p=p)


def synth_catalogue(seed, rho_conf=0.05, m=0.6, w=0.05, length_factor=1, providers=(4, 6), cue=0.4, ext_bias=1.0, annotate_extra=0.0, dup_paraphrase=0.0):
    """Returns (tools, servers). Every tool has ground truth; observable fields carry noisy cues only."""
    rng = np.random.default_rng(seed)
    specs = []                                    # (caps, role, dup_group)
    gid = 0
    for cap in CAPS:
        k = int(rng.integers(providers[0], providers[1] + 1))
        for _ in range(k):
            caps = [cap]
            for a, b in PAIRS:
                if cap == a and rng.random() < 0.25:
                    caps.append(b)
            specs.append((caps, str(_role_for(cap, rng, ext_bias)), None))
        if rng.random() < rho_conf * 4:           # rho_conf = share of capabilities with a near-duplicate pair (x4: 2 tools/pair, ~4 providers)
            gid += 1
            r1 = str(_role_for(cap, rng, ext_bias)); r2 = str(_role_for(cap, rng, ext_bias))
            specs.append(([cap], r1, gid)); specs.append(([cap], r2, gid))
    tools = []
    dup_text = {}
    for i, (caps, role, dup) in enumerate(specs):
        src = set().union(*[CAPS[c][2] for c in caps])
        accs = [CAPS[c][3] for c in caps]
        acc = set(ATT) if any(a is None for a in accs) or rng.random() < 0.3 else set().union(*accs)
        p_out = {"INS": 0.05, "PRO": 0.2, "EXT": 0.6, "PUB": 1.0}[role]
        out = int(rng.random() < p_out)
        if out and role == "INS":
            out = int(rng.random() < 0.5)
        prod = set().union(*[CAPS[c][4] for c in caps]); cons = set().union(*[CAPS[c][5] for c in caps])
        vendor = str(rng.choice(VENDORS))
        kw = " and ".join(paraphrase(CAPS[c][0], rng) for c in caps)
        if dup is not None and dup in dup_text:
            if dup_paraphrase > 0:                    # A2: lower similarity between near-duplicates
                desc = paraphrase(dup_text[dup], rng, dup_paraphrase)
            else:
                desc = dup_text[dup].replace("quickly", "easily") if "quickly" in dup_text[dup] else dup_text[dup] + " Simple."
        else:
            t = TEMPLATES[int(rng.integers(len(TEMPLATES)))]
            desc = t.format(kw=kw)
            desc = desc[0].upper() + desc[1:]
            if dup is None:
                desc = desc + f" Provided by {vendor.capitalize()}."
            for a in sorted(src):
                if rng.random() < cue:
                    desc += " " + SRC_CUES[a]
            if out and rng.random() < cue:
                desc += " " + OUT_CUES[int(rng.integers(len(OUT_CUES)))]
            if not out and rng.random() < 0.08:
                desc += " " + OUT_CUES[int(rng.integers(len(OUT_CUES)))]
            if dup is not None:
                dup_text[dup] = desc
        desc += FILLER * (max(1, length_factor) - 1)
        params = []
        for a in sorted(acc if acc != set(ATT) else set()):
            params.append(str(rng.choice(PARAM_FOR[a])))
        if acc == set(ATT):
            params.append(str(rng.choice(FREE_PARAMS)))
        if out and rng.random() < cue:
            params.append(str(rng.choice(OUT_PARAMS)))
        params = sorted(set(params))
        name = (f"{caps[0]}" if dup is not None else f"{vendor}_{caps[0]}") if rng.random() < 0.5 else f"{vendor}_{paraphrase(caps[0].replace('_', ' '), rng, 0.9).replace(' ', '_')}"
        truth = {f"src_{a}": int(a in src) for a in ATT}; truth["out"] = out
        hints = {}
        for l in LABELS:
            if rng.random() < 1 - m:
                hints[l] = truth[l] if rng.random() >= w else 1 - truth[l]
            else:
                hints[l] = truth[l] if rng.random() < annotate_extra else None   # E2: providers publish correct annotations
        lat, price, rel = _qos(rng, role)
        tdef = {"name": name, "description": desc, "inputSchema": {"type": "object", "properties": {p: {"type": "string"} for p in params}}}
        tools.append(dict(id=f"s{i}", name=name, role=role, desc=desc, params=params, true_caps=set(caps), truth=truth,
                          acc=acc, prod=prod, cons=cons, act=CAPS[caps[0]][6], hints=hints, lat=lat, price=price, rel=rel,
                          cost=approx_tokens(tdef), synthetic=True, dup=dup, domains=set().union(*[set(CAPS[c][1] or DOMAINS) for c in caps])))
    # servers: chunk tools of the same role
    servers = {}
    by_role = {r: [t for t in tools if t["role"] == r] for r in ROLES}
    sid = 0
    for r, lst in by_role.items():
        rng.shuffle(lst)
        j = 0
        while j < len(lst):
            size = int(rng.integers(2, 9))
            name = f"{r.lower()}-{rng.choice(VENDORS)}-{sid}"
            servers[name] = dict(name=name, role=r, avail=float(rng.uniform(0.90, 0.999)), preamble=int(rng.integers(30, 250)), synthetic=True)
            for t in lst[j:j + size]:
                t["server"] = name
            j += size; sid += 1
    # a duplicate pair must sit on two different servers
    for g in {t["dup"] for t in tools if t["dup"] is not None}:
        pair = [t for t in tools if t["dup"] == g]
        if len(pair) == 2 and pair[0]["server"] == pair[1]["server"]:
            others = [s for s in servers if s != pair[0]["server"] and servers[s]["role"] == pair[1]["role"]]
            if not others:
                others = [s for s in servers if s != pair[0]["server"]]
            pair[1]["server"] = others[int(rng.integers(len(others)))]; pair[1]["role"] = servers[pair[1]["server"]]["role"]
    return tools, servers


PUB_RX = re.compile(r"search|fetch|brows|scrap|crawl", re.I)


def real_catalogue(path, seed, max_tools=None):
    """MCP-tools: no annotations, no ground truth. Roles: PUB for web access, EXT otherwise."""
    rng = np.random.default_rng(seed + 7919)
    S = json.load(open(path, encoding="utf-8"))
    tools, servers = [], {}
    for s in S:
        role = "PUB" if PUB_RX.search(s["name"] + " " + s.get("description", "")) else "EXT"
        sname = "real:" + s["name"]
        servers[sname] = dict(name=sname, role=role, avail=float(rng.uniform(0.90, 0.999)),
                              preamble=approx_tokens({"name": s["name"], "description": s.get("description", "")}), synthetic=False)
        for t in s["tools"]:
            params = sorted(t.get("parameter", {}).keys())
            lat, price, rel = _qos(rng, role)
            tdef = {"name": t["name"], "description": t["description"], "parameters": t.get("parameter", {})}
            tools.append(dict(id=f"r{len(tools)}", name=t["name"], role=role, server=sname, desc=t["description"], params=params,
                              true_caps=None, truth=None, acc=set(ATT), prod=set(), cons=set(), act="search" if role == "PUB" else "illustrate",
                              hints={l: None for l in LABELS}, lat=lat, price=price, rel=rel, cost=approx_tokens(tdef),
                              synthetic=False, dup=None, domains=set(DOMAINS)))
    if max_tools and len(tools) > max_tools:
        idx = rng.choice(len(tools), max_tools, replace=False)
        tools = [tools[i] for i in sorted(idx)]
        keep = {t["server"] for t in tools}
        servers = {k: v for k, v in servers.items() if k in keep}
    return tools, servers


# ---------------------------------------------------------------- requests and workflows (generator contract)
def _closure(domain, targets):
    pre = dict(DOMAIN_CONCEPTS[domain]); seen = []
    def visit(c):
        for p in pre[c]:
            visit(p)
        if c not in seen:
            seen.append(c)
    for c in targets:
        visit(c)
    return seen


def make_task(rng, domain, ty, concepts, extra=None):
    allowed = [c for c in TYPE_CAPS[ty] if CAPS[c][1] is None or domain in CAPS[c][1]]
    core = CORE_CAP.get((ty, domain)) or CORE_CAP.get((ty, None))
    R = {core} if core in allowed else set()
    n = int(rng.integers(1, 5)) if extra is None else extra
    pool = [c for c in allowed if c not in R]
    while len(R) < n and pool:
        R.add(pool.pop(int(rng.integers(len(pool)))))
    return dict(type=ty, domain=domain, concepts=list(concepts), level=LEVEL[ty], R=set(R),
                D=set(DEFAULT_D[ty]), asm=ASM[ty])


def generate_workflow(rng, domain, targets, known=0):
    """Rule-based generator satisfying contract (a)-(d): acyclic, prerequisites upstream, admissible capabilities, typed hand-offs."""
    full = _closure(domain, targets)[known:]
    keep = set(full[-3:]) | set(targets)                     # never drop a target: prerequisites stay upstream
    chain = [c for c in full if c in keep]
    tasks = []
    for c in chain:
        if c in targets or rng.random() < 0.5:
            tasks.append(make_task(rng, domain, "acquisition", [c]))
        tasks.append(make_task(rng, domain, "practice", [c]))
    tasks.append(make_task(rng, domain, "formative", chain[-2:]))
    end = rng.choice(["summative", "production", "discussion"], p=[0.5, 0.3, 0.2])
    tasks.append(make_task(rng, domain, str(end), targets))
    tasks = tasks[-8:]
    edges = []
    for i in range(len(tasks) - 1):
        up, dn = tasks[i], tasks[i + 1]
        ho = set()
        for art in HANDOFF_ATTRS:
            if any(art in CAPS[c][4] for c in up["R"]) and any(art in CAPS[c][5] for c in dn["R"]):
                ho.add(art)
        edges.append((i, i + 1, ho))
    return dict(domain=domain, targets=list(targets), tasks=tasks, edges=edges)


def generate_requests(seed, n_per_domain=34):
    rng = np.random.default_rng(seed + 104729)
    reqs = []
    for d in DOMAINS:
        names = [c for c, _ in DOMAIN_CONCEPTS[d]]
        for _ in range(n_per_domain):
            k = int(rng.integers(1, 3))
            targets = [str(x) for x in rng.choice(names[2:], k, replace=False)]
            reqs.append(dict(domain=d, targets=targets, known=int(rng.integers(0, 2))))
    return reqs


def inherited_attrs(wf, i):
    """delta^-(v): attributes carried by hand-off artefacts into task i (isolated contexts)."""
    out = set()
    for (a, b, ho) in wf["edges"]:
        if b == i:
            for art in ho:
                out |= HANDOFF_ATTRS[art]
    return out


def sample_tasks(seed, n=300):
    """Task-level benchmark: n tasks stratified by pedagogical type (n/6 per type), taken from generated workflows."""
    rng = np.random.default_rng(seed + 15485863)
    per = n // len(TYPES); bucket = {t: [] for t in TYPES}
    reqs = generate_requests(seed, n_per_domain=60)
    for rq in reqs:
        wf = generate_workflow(rng, rq["domain"], rq["targets"], rq["known"])
        for i, t in enumerate(wf["tasks"]):
            tt = dict(t); tt["delta_in"] = inherited_attrs(wf, i)
            tt["ho_out"] = set().union(*[ho for (a, b, ho) in wf["edges"] if a == i]) if any(a == i for a, _, _ in wf["edges"]) else set()
            tt["ho_in"] = set().union(*[ho for (a, b, ho) in wf["edges"] if b == i]) if any(b == i for _, b, _ in wf["edges"]) else set()
            bucket[t["type"]].append(tt)
    out = []
    for ty in TYPES:
        lst = bucket[ty]
        while len(lst) < per:                      # rare types: top up with fresh tasks of that type
            d = DOMAINS[int(rng.integers(3))]
            tt = make_task(rng, d, ty, [DOMAIN_CONCEPTS[d][int(rng.integers(len(DOMAIN_CONCEPTS[d])))][0]])
            tt["delta_in"] = set(); tt["ho_out"] = set(); tt["ho_in"] = set(); lst.append(tt)
        idx = rng.choice(len(lst), per, replace=False)
        out += [lst[i] for i in idx]
    return out


def archetypes(seed, source="esco"):
    """Archetypes = (pedagogical type, skill). Skills and demand weights from ESCO when available, else the domain concepts."""
    rng = np.random.default_rng(seed + 32452843)
    if source == "esco":
        try:
            from .esco import domain_skills, attach_concepts
            arch = []
            for d in DOMAINS:
                for sk in attach_concepts(d, domain_skills(d), [c for c, _ in DOMAIN_CONCEPTS[d]]):
                    for ty in TYPES:
                        t = make_task(rng, d, ty, [sk["concept"]], extra=2)
                        t.update(delta_in=set(), ho_out=set(), ho_in=set(), dem=sk["demand"], skill=sk["label"], skill_uri=sk["uri"])
                        arch.append(t)
            return arch
        except FileNotFoundError:
            pass
    arch = []
    for d in DOMAINS:
        for c, _ in DOMAIN_CONCEPTS[d]:
            for ty in TYPES:
                t = make_task(rng, d, ty, [c], extra=2)
                t["delta_in"] = set(); t["ho_out"] = set(); t["ho_in"] = set()
                t["dem"] = float(rng.zipf(2.0))
                arch.append(t)
    return arch
