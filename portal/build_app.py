"""Builds the single-file portal: inlines portal/bundle.json, portal/engine.js, Chart.js (portal/assets/chart.umd.js, v4.4.4) and
the figures of the paper (portal/assets/fig_*.png) into portal/app.html, and writes portal/index.html and docs/index.html.
Set EMERKIT_REPO_URL to show the GitHub link on the About page. The file opens directly from disk."""
import os, base64, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P = lambda *a: os.path.join(ROOT, "portal", *a)
rd = lambda p: open(p, encoding="utf-8").read()
figs = {os.path.basename(f)[4:-4]: "data:image/png;base64," + base64.b64encode(open(f, "rb").read()).decode() for f in glob.glob(P("assets", "fig_*.png"))}
safe = lambda s: s.replace("</script", "<\\/script")
html = (rd(P("app.html")).replace("__BUNDLE__", rd(P("bundle.json")).replace("</", "<\\/")).replace("__ENGINE__", safe(rd(P("engine.js"))))
        .replace("__CHARTJS__", safe(rd(P("assets", "chart.umd.js")))).replace("__FIGS__", json.dumps(figs))
        .replace("__REPO_URL__", os.environ.get("EMERKIT_REPO_URL", "__REPO_URL__")))
for d in ["portal", "docs"]:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True); open(os.path.join(ROOT, d, "index.html"), "w", encoding="utf-8").write(html)
open(os.path.join(ROOT, "docs", ".nojekyll"), "w").close()
print("index.html", len(html.encode()) // 1024, "KB")
