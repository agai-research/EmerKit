"""Audit of real MCP tools, step 1: static analysis of server source code (GitHub archives via codeload, no API token needed).
Evidence per server: outbound network calls (out), storage of user data, identifiers and profile fields handled (src labels).
Step 2 (run_llm.py audit): an LLM panel reviews the handler code of each sampled tool; step 3: the authors check a sample.
Usage: python scripts/audit_static.py N_SERVERS [SEED]"""
import io, json, os, re, sys, tarfile, urllib.request
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from emerkit.sources import toucan_catalogue

NET = re.compile(r"\b(requests\.(get|post|put|patch)|httpx\.|aiohttp|urllib\.request|fetch\(|axios|got\(|node-fetch|http\.client|"
                 r"WebClient|HttpClient|reqwest|net/http)\b")
URL = re.compile(r"https?://(?!localhost|127\.0\.0\.1)[\w.-]+\.[a-z]{2,}")
STORE = re.compile(r"\b(open\([^)]*['\"]w|writeFile|sqlite3|INSERT INTO|\.save\(|redis|mongodb|s3\.put|putObject)\b", re.I)
ID_RX = re.compile(r"\b(user_?id|student_?id|email|username|account)\b", re.I)
PROF_RX = re.compile(r"\b(profile|preferences|settings|language|level)\b", re.I)
PERF_RX = re.compile(r"\b(score|grade|result|mark)\b", re.I)
BEH_RX = re.compile(r"\b(history|activity|log|usage|events?)\b", re.I)
SRC_EXT = (".py", ".js", ".ts", ".mjs", ".go", ".rs", ".java", ".kt", ".rb")


def fetch_repo(url, max_bytes=30_000_000):
    m = re.match(r"https?://github\.com/([^/]+)/([^/#?]+)", url or "")
    if not m:
        return None
    owner, repo = m.group(1), m.group(2).removesuffix(".git")
    for ref in ("HEAD", "refs/heads/main", "refs/heads/master"):
        try:
            with urllib.request.urlopen(f"https://codeload.github.com/{owner}/{repo}/tar.gz/{ref}", timeout=40) as r:
                data = r.read(max_bytes)
            return tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        except Exception:
            continue
    return None


def scan(tf, tool_names):
    ev = dict(files=0, net=0, urls=set(), store=0, tools={n: {"found": False, "net": 0, "ID": 0, "PROF": 0, "PERF": 0, "BEH": 0} for n in tool_names})
    for m in tf.getmembers():
        if not m.isfile() or not m.name.endswith(SRC_EXT) or "node_modules" in m.name or "/test" in m.name.lower():
            continue
        try:
            code = tf.extractfile(m).read(400_000).decode("utf-8", "ignore")
        except Exception:
            continue
        ev["files"] += 1
        ev["net"] += len(NET.findall(code)); ev["urls"] |= set(URL.findall(code)); ev["store"] += len(STORE.findall(code))
        for n in tool_names:
            k = code.find(n)
            if k >= 0:
                window = code[k: k + 4000]                              # handler body approximated by the following code
                t = ev["tools"][n]; t["found"] = True; t["net"] += len(NET.findall(window))
                for a, rx in [("ID", ID_RX), ("PROF", PROF_RX), ("PERF", PERF_RX), ("BEH", BEH_RX)]:
                    t[a] += len(rx.findall(window))
    ev["urls"] = sorted(ev["urls"])[:50]
    return ev


def main(n_servers, seed=0):
    T, S = toucan_catalogue(seed)
    rng = np.random.default_rng(seed)
    with_repo = sorted([s for s, v in S.items() if (v.get("repository") or "").startswith("https://github.com/")])
    pick = list(rng.choice(with_repo, min(n_servers, len(with_repo)), replace=False))
    out_path = os.path.join(ROOT, "data", "audit", "static_evidence.jsonl"); os.makedirs(os.path.dirname(out_path), exist_ok=True)
    done = {json.loads(l)["server"] for l in open(out_path)} if os.path.exists(out_path) else set()
    for s in pick:
        if s in done:
            continue
        names = [t["name"] for t in T if t["server"] == s]
        tf = fetch_repo(S[s]["repository"])
        rec = dict(server=s, repository=S[s]["repository"], tools=names, fetched=tf is not None)
        if tf is not None:
            try:
                rec.update(scan(tf, names))
            except (EOFError, OSError, tarfile.TarError) as e:          # truncated or corrupt archive
                rec.update(fetched=False, error=type(e).__name__)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(s, rec["fetched"], rec.get("files"), rec.get("net"), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
