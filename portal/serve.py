"""Runs the EmerKit portal on your computer: python portal/serve.py, then open http://localhost:8000.
Serves docs/index.html and relays, from your computer, the requests that browsers may block across sites:
  POST /api/anthropic  -> https://api.anthropic.com/v1/messages (key from the request or from ANTHROPIC_API_KEY)
  POST /api/openai     -> <base>/chat/completions of an OpenAI-compatible server, for example a local Ollama
                          (base URL in the x-target-base header, default http://localhost:11434/v1)
  GET  /api/registry   -> https://registry.modelcontextprotocol.io/v0.1/servers (official MCP Registry)
Nothing is stored; the relay listens on localhost only. Python standard library only (certifi is used when installed)."""
import http.server, json, os, sys, ssl, shutil, subprocess, urllib.request, urllib.parse, urllib.error
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs"); PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


def _context():
    """TLS context: certifi when installed (python.org builds on macOS often lack root certificates), else the system default."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


CTX = _context()


def fetch(url, method="GET", headers=None, body=None, timeout=60):
    """Returns (status, bytes). Falls back to the system curl when Python cannot verify certificates."""
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, ssl.SSLError) as e:
        reason = getattr(e, "reason", e)
        if url.startswith("https://") and shutil.which("curl") and isinstance(reason, ssl.SSLError):
            cmd = ["curl", "-sS", "-m", str(timeout), "-X", method, "-w", "\n%{http_code}", url]
            for k, v in (headers or {}).items():
                cmd += ["-H", f"{k}: {v}"]
            if body is not None:
                cmd += ["--data-binary", "@-"]
            out = subprocess.run(cmd, input=body, capture_output=True, timeout=timeout + 5)
            if out.returncode == 0:
                data, _, code = out.stdout.rpartition(b"\n")
                return int(code or 502), data
            raise RuntimeError(out.stderr.decode(errors="ignore").strip() or str(e))
        raise RuntimeError(str(reason))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=ROOT, **k)

    def _send(self, code, body, ctype="application/json"):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/health":
            return self._send(200, json.dumps({"ok": True, "env_key": bool(os.environ.get("ANTHROPIC_API_KEY")), "relay": "emerkit"}).encode())
        if self.path.startswith("/api/registry"):
            q = urllib.parse.urlparse(self.path).query
            try:
                code, data = fetch(os.environ.get("EMERKIT_REGISTRY_BASE", "https://registry.modelcontextprotocol.io").rstrip("/") + "/v0.1/servers?" + q, headers={"Accept": "application/json", "User-Agent": "EmerKit-portal/1.0"}, timeout=30)
                return self._send(code, data)
            except Exception as e:
                return self._send(502, json.dumps({"error": f"the relay could not reach the official registry ({e})"}).encode())
        return super().do_GET()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path == "/api/anthropic":
            key = self.headers.get("x-api-key") or os.environ.get("ANTHROPIC_API_KEY", "")
            try:
                code, data = fetch("https://api.anthropic.com/v1/messages", "POST",
                                   {"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"}, body, timeout=85)
                return self._send(code, data)
            except Exception as e:
                return self._send(502, json.dumps({"error": {"message": f"the relay could not reach api.anthropic.com ({e})"}}).encode())
        if self.path == "/api/openai":
            base = (self.headers.get("x-target-base") or "http://localhost:11434/v1").rstrip("/")
            if not base.startswith(("http://", "https://")):
                return self._send(400, json.dumps({"error": {"message": "invalid base URL"}}).encode())
            hdr = {"content-type": "application/json"}
            if self.headers.get("x-api-key"):
                hdr["authorization"] = "Bearer " + self.headers["x-api-key"]
            try:
                code, data = fetch(base + "/chat/completions", "POST", hdr, body, timeout=85)
                return self._send(code, data)
            except Exception as e:
                return self._send(502, json.dumps({"error": {"message": f"the relay could not reach {base} ({e}); is the model server running?"}}).encode())
        return self._send(404, b"{}")

    def log_message(self, *a): pass


if __name__ == "__main__":
    print(f"EmerKit portal: http://localhost:{PORT}  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
