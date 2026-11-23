"""Real catalogues. Toucan server metadata (Xu et al., 2025; github.com/TheAgentArk/Toucan, mcp_servers/*.json): full tool
schemas, MCP annotations, categories and measured success rates of remote MCP servers. MCP-tools is loaded by data.real_catalogue."""
import json, re, tarfile, os
import numpy as np
from .vocab import ATT, DOMAINS
from .data import LABELS, approx_tokens, _qos

HERE = os.path.dirname(os.path.abspath(__file__))
TOUCAN = os.path.join(HERE, "..", "data", "sources", "toucan_mcp_servers.tgz")
PUB_CAT = re.compile(r"web search|browser|search", re.I)
EDU_RX = re.compile(r"educat|learn|course|student|teach|tutor|quiz|flashcard|school|exam|lesson", re.I)
FREE = re.compile(r"query|text|content|prompt|message|body|input|code|data|payload|description|note", re.I)


def _rate(x):
    try:
        x = str(x).strip(); return float(x.strip("%")) / 100 if "%" in x else float(x)
    except ValueError:
        return None


def toucan_catalogue(seed=0, path=TOUCAN, max_servers=None):
    rng = np.random.default_rng(seed + 4049)
    tools, servers = [], {}
    with tarfile.open(path) as tf:
        members = sorted([m for m in tf.getmembers() if m.name.endswith(".json")], key=lambda m: m.name)
        for m in members:
            d = json.load(tf.extractfile(m)); md = d["metadata"]; resp = md.get("remote_server_response") or {}
            if not resp.get("is_success") or not resp.get("tools"):
                continue
            lab = d.get("labels", {}); info = md.get("server_info_crawled", {})
            cats = " ".join([lab.get("primary_label") or ""] + list(lab.get("secondary_labels") or []))
            role = "PUB" if PUB_CAT.search(cats) else "EXT"
            sname = "toucan:" + md["server_name"]
            rel_srv = _rate(info.get("success_rate"))
            servers[sname] = dict(name=sname, role=role, avail=float(rng.uniform(0.90, 0.999)),
                                  preamble=approx_tokens({"name": md["server_name"], "overview": info.get("overview", "")}), synthetic=False,
                                  category=lab.get("primary_label"), education=bool(EDU_RX.search(cats + " " + str(info.get("overview", "")))),
                                  repository=info.get("repository_url"), usage=info.get("usage_count"))
            for t in resp["tools"]:
                props = (t.get("input_schema") or {}).get("properties") or {}
                props = props if isinstance(props, dict) else {}
                ann = t.get("annotations") or {}
                hints = {l: None for l in LABELS}
                if "openWorldHint" in ann:
                    hints["out"] = int(bool(ann["openWorldHint"]))
                lat, price, rel = _qos(rng, role)
                if rel_srv is not None:
                    rel = float(np.clip(rel_srv, 0.01, 0.999))
                acc = set() if not props else (set(ATT) if any(FREE.search(k) for k in props) or any((v.get("type") if isinstance(v, dict) else None) in (None, "string", "object", "array") for v in props.values()) else {"ID"})
                tdef = {"name": t["name"], "description": t.get("description", ""), "inputSchema": t.get("input_schema", {})}
                tools.append(dict(id=f"t{len(tools)}", name=t["name"], role=role, server=sname, desc=t.get("description") or "", params=sorted(props),
                                  true_caps=None, truth=None, acc=acc, prod=set(), cons=set(), act="search" if role == "PUB" else "illustrate",
                                  hints=hints, lat=lat, price=price, rel=rel, cost=approx_tokens(tdef), synthetic=False, dup=None,
                                  domains=set(DOMAINS), annotations=ann, category=lab.get("primary_label"),
                                  labels=[x for x in [lab.get("primary_label")] + list(lab.get("secondary_labels") or []) if x]))
            if max_servers and len(servers) >= max_servers:
                break
    return tools, servers
