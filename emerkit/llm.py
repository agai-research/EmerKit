"""Model-dependent components. NOT executed in the pilot environment (no API access); run them in Claude Code.

Client: EMERKIT_PROVIDER = anthropic | openai (any OpenAI-compatible endpoint: vLLM, Ollama, ...),
EMERKIT_MODEL = model name, EMERKIT_BASE_URL for OpenAI-compatible servers. Every call is logged to llm_log.jsonl.
"""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor
_LOG_LOCK = threading.Lock()
import numpy as np
from .vocab import CAPS, TYPE_CAPS, TYPES, DOMAIN_CONCEPTS, ASM, LEVEL, DEFAULT_D


class LLMClient:
    def __init__(self, provider=None, model=None, temperature=0.0, log="llm_log.jsonl"):
        self.provider = provider or os.environ.get("EMERKIT_PROVIDER", "anthropic")
        self.model = model or os.environ.get("EMERKIT_MODEL")
        self.temperature, self.log = temperature, log
        if self.provider == "anthropic":
            import anthropic; self.c = anthropic.Anthropic()
        else:
            import openai; self.c = openai.OpenAI(base_url=os.environ.get("EMERKIT_BASE_URL"), api_key=os.environ.get("OPENAI_API_KEY", "none"))

    def __call__(self, system, user, max_tokens=1024):
        t0 = time.time()
        if os.environ.get("EMERKIT_NO_THINK"):                                         # Qwen3 soft switch: answer without thinking
            user = user + " /no_think"
        if self.provider == "anthropic":
            for attempt in range(5):                      # retry on rate limits and transient errors
                try:
                    r = self.c.messages.create(model=self.model, max_tokens=max_tokens, system=system,
                                               messages=[{"role": "user", "content": user}])
                    break
                except Exception as e:
                    if attempt == 4: raise
                    time.sleep(2 ** attempt)
            text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
            usage = dict(inp=r.usage.input_tokens, out=r.usage.output_tokens)
        else:
            r = self.c.chat.completions.create(model=self.model, temperature=self.temperature, max_tokens=max_tokens,
                                               messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            text = r.choices[0].message.content or ""; usage = dict(inp=r.usage.prompt_tokens, out=r.usage.completion_tokens)
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()      # reasoning models (Qwen3) emit their thoughts
        with _LOG_LOCK, open(self.log, "a") as f:
            f.write(json.dumps(dict(model=self.model, system=system[:200], user=user[:4000], output=text, sec=time.time() - t0, **usage)) + "\n")
        return text


def pmap(fn, items, workers=None):
    """Parallel map over independent model calls (the clients are thread-safe); keeps the input order."""
    workers = workers or int(os.environ.get("EMERKIT_WORKERS", "8"))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items))


def parse_json(text):
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    return json.loads(m.group(0)) if m else None


def tool_def(t):
    return {"name": f"{t['name']}__{t['id']}", "description": t["desc"], "inputSchema": {"type": "object", "properties": {p: {"type": "string"} for p in t["params"]}}}


# ------------------------------------------------------------------------------------------------ Compositio Prompto (adapted)
CP_SYSTEM = ("You compose services. Given a task in natural language, the specifications of the available services and the expected "
             "output schema, return only a JSON object that satisfies the schema.")
CP_SCHEMA = {"tasks": [{"type": "one of " + "|".join(TYPES), "concepts": ["concept"], "capabilities": ["capability"], "tools": ["tool name"]}]}


def compositio_prompto(client, request, tools, cand_idx, tries=3, scorer=None):
    """Adaptation of Pesl et al. (ICSOC 2024): service specifications + task + I/O schema -> composition (best of three).
    Specifications are the MCP definitions of the same candidate set given to the other methods (an MCP catalogue does not fit a prompt)."""
    specs = json.dumps([tool_def(tools[i]) for i in cand_idx], ensure_ascii=False)
    user = (f"Task: a learner asks to learn {', '.join(request['targets'])} ({request['domain']}). Build a learning workflow and, for each task, "
            f"the tools an AI agent should load.\nCapabilities: {', '.join(CAPS)}\nServices:\n{specs}\nOutput schema: {json.dumps(CP_SCHEMA)}")
    best, best_s = None, -np.inf
    for _ in range(tries):
        out = parse_json(client(CP_SYSTEM, user, max_tokens=6000))
        if not out or "tasks" not in out:
            continue
        s = scorer(out) if scorer else 0.0
        if s > best_s:
            best, best_s = out, s
    return best


def resolve_names(comp, tools):
    by = {f"{t['name']}__{t['id']}": i for i, t in enumerate(tools)}
    return [[by[n] for n in t.get("tools", []) if n in by] for t in comp.get("tasks", [])]


# ------------------------------------------------------------------------------------------------ selection probes (A5, A6)
PROBE_SYSTEM = "You are an AI agent. Choose the single tool you would call for the request. Answer with the tool name only."


def probe(client, task_text, toolkit, tools):
    defs = json.dumps([tool_def(tools[i]) for i in toolkit], ensure_ascii=False)
    out = client(PROBE_SYSTEM, f"Available tools:\n{defs}\n\nRequest: {task_text}", max_tokens=400).strip()
    ids = {tools[i]["id"]: i for i in toolkit}
    for m in re.finditer(r"__([A-Za-z]?\d+)", out):          # the id suffix identifies the tool even if the name is altered
        if m.group(1) in ids:
            return ids[m.group(1)]
    names = {tools[i]["name"]: i for i in toolkit}
    hits = [i for n, i in names.items() if n in out]
    return hits[0] if len(hits) == 1 else None


def probe_design(rng, prep, n_per_cell=40, ks=(2, 4, 6, 8, 12), loads=(0.25, 0.5, 0.75, 1.0, 1.25), B=2000):
    """Cells: toolkit size x confusable distractor present x load ratio. A choice is correct when the chosen tool truly provides the
    probed capability; the confusable distractor is the non-provider whose description is closest to the target's."""
    T = prep.tools; design = []
    caps = [r for r in CAPS if any(t["true_caps"] and r in t["true_caps"] for t in T)]
    for k in ks:
        for dup in (0, 1):
            for lr in loads:
                for _ in range(n_per_cell):
                    r = caps[int(rng.integers(len(caps)))]
                    prov = [i for i, t in enumerate(T) if t["true_caps"] and r in t["true_caps"]]
                    target = int(rng.choice(prov))
                    others = [i for i in range(len(T)) if T[i]["true_caps"] and r not in T[i]["true_caps"]]
                    kit = [target] + list(rng.choice(others, k - 1 - dup, replace=False))
                    if dup:                                  # confusable distractor: the most similar tool that lacks the capability
                        sims = [(prep.desc_sim(target, i), i) for i in others if i not in kit]
                        kit.append(int(max(sims)[1]))
                    design.append(dict(cap=r, target=target, toolkit=kit, k=k, dup=dup, load_ratio=lr, pad_tokens=max(0, int(lr * B) - sum(T[i]["cost"] for i in kit))))
    return design


# ------------------------------------------------------------------------------------------------ LLM workflow generator and capability extraction
GEN_SYSTEM = "You design learning workflows. Return only JSON."


def llm_generate_workflow(client, request):
    from .data import _closure
    closure = _closure(request["domain"], request["targets"])[-12:]           # prerequisite order, most relevant concepts
    caps = {ty: [c for c in TYPE_CAPS[ty] if CAPS[c][1] is None or request["domain"] in CAPS[c][1]] for ty in TYPES}
    user = (f"Learner request: learn {', '.join(request['targets'])} in {request['domain']}.\nConcepts you may use, prerequisites first: {closure}.\n"
            f"Task types: {TYPES}.\nCapabilities allowed for each type in this domain: {json.dumps(caps)}.\n"
            f"Return {{\"tasks\": [{{\"type\": .., \"concepts\": [..], \"capabilities\": [..]}}]}} with 3 to 8 tasks, ordered so that every "
            f"concept comes after its prerequisites, covering every requested concept, and using only allowed capabilities.")
    out = parse_json(client(GEN_SYSTEM, user, max_tokens=4000)) or {"tasks": []}
    tasks = [dict(type=t["type"], domain=request["domain"], concepts=list(t.get("concepts", [])), level=LEVEL[t["type"]], R=set(t.get("capabilities", [])),
                  D=set(DEFAULT_D[t["type"]]), asm=ASM[t["type"]]) for t in out.get("tasks", []) if isinstance(t, dict) and t.get("type") in TYPES]
    return dict(domain=request["domain"], targets=request["targets"], tasks=tasks, edges=[(i, i + 1, set()) for i in range(len(tasks) - 1)])


def llm_capabilities(client, tool):
    user = (f"Tool: {tool['name']}: {tool['desc']} Parameters: {tool['params']}\nWhich of these capabilities does it provide? "
            f"{list(CAPS)}. Answer with a JSON list (possibly empty).")
    out = parse_json(client("Return only JSON.", user, max_tokens=1000))
    return {c for c in (out or []) if c in CAPS}


# ------------------------------------------------------------------------------------------------ Toucan (A7)
def toucan_sample(config="Kimi-K2", n=20000, seed=42):
    from datasets import load_dataset
    ds = load_dataset("Agent-Ark/Toucan-1.5M", config, split="train", streaming=True).shuffle(seed=seed, buffer_size=5000)
    return [row for _, row in zip(range(n), ds)]
