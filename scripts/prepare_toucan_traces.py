"""Converts a Toucan sample (rows of Agent-Ark/Toucan-1.5M) into usage traces on the Toucan server catalogue.
Each trace keeps the question, the (server, tool) pairs actually called, the quality scores and the prompt group (row_id).
Usage: python scripts/prepare_toucan_traces.py data/traces/toucan_sample.jsonl.gz"""
import gzip, json, os, re, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from emerkit.sources import toucan_catalogue

def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def main(path):
    T, S = toucan_catalogue(0)
    keys = {(t["server"][len("toucan:"):], t["name"]) for t in T}
    stats = collections.Counter(); out = []
    for line in gzip.open(path, "rt", encoding="utf-8"):
        r = json.loads(line); md = json.loads(r["metadata"]); msgs = json.loads(r["messages"])
        servers = [s["server_name"] for s in md.get("mcp_servers", [])]
        prefixes = sorted(((slug(s), s) for s in servers), key=lambda x: -len(x[0]))
        called = []
        for m in msgs:
            if m.get("role") != "assistant": continue
            fcs = [m["function_call"]] if m.get("function_call") else [c.get("function", c) for c in (m.get("tool_calls") or [])]
            for fc in fcs:
                name = fc.get("name", ""); stats["calls"] += 1
                for p, s in prefixes:
                    if name.startswith(p + "-"):
                        tool = name[len(p) + 1:]
                        if (s, tool) in keys:
                            called.append(f"{s}::{tool}"); stats["mapped"] += 1
                        else:
                            stats["unknown_tool"] += 1
                        break
                else:
                    stats["unknown_server"] += 1
        rq = json.loads(r.get("response_quality_assessment") or "{}")
        out.append(dict(query=r["question"], tools=sorted(set(called)), servers=servers, group=md.get("row_id"),
                        overall=rq.get("overall_score"), desired_used=rq.get("desired_tools_used_percentage"),
                        success=bool(rq.get("overall_score") and rq["overall_score"] >= 4 and (rq.get("desired_tools_used_percentage") or 0) >= 0.5)))
    dst = os.path.join(ROOT, "data", "traces", "toucan_traces.jsonl")
    with open(dst, "w") as f:
        for x in out: f.write(json.dumps(x) + "\n")
    used = [x for x in out if x["tools"]]
    print(dict(stats), "| traces", len(out), "with mapped tools", len(used), "| successful", sum(x["success"] for x in used),
          "| tools per trace (mean)", round(sum(len(x["tools"]) for x in used) / max(1, len(used)), 2),
          "| distinct tools", len({t for x in used for t in x["tools"]}), "| servers", len({s for x in used for s in x["servers"]}))

if __name__ == "__main__":
    main(sys.argv[1])
