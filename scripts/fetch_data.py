"""Downloads the third-party data that cannot be redistributed or that is too large: LectureBank 2.0 (no licence file),
Toucan server metadata (refreshed), and checks MCP-tools. Run once after cloning: python scripts/fetch_data.py"""
import io, os, shutil, tarfile, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SRC = os.path.join(ROOT, "data", "sources"); os.makedirs(SRC, exist_ok=True)

def archive(repo, branch):
    with urllib.request.urlopen(f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}", timeout=120) as r:
        return tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")

lb = os.path.join(SRC, "lecturebank2")
if not os.path.exists(os.path.join(lb, "final_new_annotation.csv")):
    tf = archive("Yale-LILY/LectureBank", "master"); os.makedirs(lb, exist_ok=True)
    for m in tf.getmembers():
        if m.isfile() and "/LB-Paper/LectureBank2/data/" in m.name and m.name.count("/") == 4:
            with open(os.path.join(lb, os.path.basename(m.name)), "wb") as f:
                f.write(tf.extractfile(m).read())
    print("LectureBank 2.0 ->", lb)
tc = os.path.join(SRC, "toucan_mcp_servers.tgz")
if not os.path.exists(tc):
    tf = archive("TheAgentArk/Toucan", "main")
    with tarfile.open(tc, "w:gz") as out:
        for m in tf.getmembers():
            if m.name.startswith("Toucan-main/mcp_servers/"):
                out.addfile(m, tf.extractfile(m) if m.isfile() else None)
    print("Toucan server metadata ->", tc)
mt = os.path.join(ROOT, "data", "mcp_tools_compact.json")
print("MCP-tools present:", os.path.exists(mt), "(download mcp_tools_with_embedding.json from the link in github.com/xfey/MCP-Zero/MCP-tools otherwise)")
