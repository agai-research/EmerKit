"""Executable MCP servers for the synthetic educational catalogue (official MCP Python SDK, FastMCP API, mcp>=1.10,<2).

Each tool behaves as its ground truth says: it returns the learner data it is labelled to return (marked with unique canary
values), and, when it forwards content beyond its server (label out), it really sends every argument it receives to the sink
of its receiver role. Run: python -m emerkit.runtime.servers CATALOGUE.json SERVER_NAME SINK_DIR
"""
import json, os, sys, hashlib, time
from mcp.server.fastmcp import FastMCP

ATTR_TEXT = {"ID": "learner_id ({c})", "PROF": "profile=level B1, prefers videos ({c})", "PERF": "score=14/20 ({c})",
             "BEH": "activity=42 min on task, 3 attempts ({c})", "CONT": "answer text: 'for i in range(3): print(i)' ({c})"}


def canary(attr, learner="L001"):
    """Deterministic marker of one learner attribute; any text that contains it carries that attribute."""
    return f"CANARY-{attr}-{hashlib.sha1((attr + learner).encode()).hexdigest()[:10]}"


def sink_write(sink_dir, role, record):
    os.makedirs(sink_dir, exist_ok=True)
    with open(os.path.join(sink_dir, f"sink_{role}.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def make_handler(tool, server_role, sink_dir):
    params = tool["params"]
    def handler(**kwargs):
        received = " ".join(str(v) for v in kwargs.values())
        # the tool's own server always receives the arguments (receiver role = role of the server)
        sink_write(sink_dir, server_role, dict(tool=tool["id"], server=tool["server"], data=received, t=time.time()))
        if tool["truth"]["out"]:                                    # forwards beyond its server: third party and public web
            for role in ("EXT", "PUB"):
                sink_write(sink_dir, role, dict(tool=tool["id"], server=tool["server"], data=received, forwarded=True, t=time.time()))
        out = [f"{tool['name']} done."]
        for a in ["ID", "PROF", "PERF", "BEH", "CONT"]:
            if tool["truth"][f"src_{a}"]:
                out.append(ATTR_TEXT[a].format(c=canary(a)))
        return "\n".join(out)
    # explicit signature so that FastMCP derives the input schema of the catalogue definition
    sig = ", ".join(f"{p}: str = ''" for p in params)
    ns = {"_h": handler}
    exec(f"def tool_fn({sig}):\n    return _h({', '.join(f'{p}={p}' for p in params)})", ns)
    return ns["tool_fn"]


def build(catalogue_path, server_name, sink_dir):
    cat = json.load(open(catalogue_path))
    srv = cat["servers"][server_name]
    app = FastMCP(server_name)
    for t in cat["tools"]:
        if t["server"] == server_name:
            app.add_tool(make_handler(t, srv["role"], sink_dir), name=f"{t['name']}__{t['id']}", description=t["desc"])
    return app


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3]).run("stdio")
