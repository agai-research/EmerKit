"""MCP gateway: exposes to the agent only the tools of the toolkit K_v of the task being enacted, rejects every other call,
and logs each call. One gateway (one agent session) per task. Run:
python -m emerkit.runtime.gateway CATALOGUE.json SINK_DIR LOG.jsonl TOOL_ID [TOOL_ID ...]"""
import json, sys, time, asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from mcp.server.fastmcp import FastMCP
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def build(catalogue_path, sink_dir, log_path, allowed):
    cat = json.load(open(catalogue_path))
    tools = {t["id"]: t for t in cat["tools"] if t["id"] in set(allowed)}
    servers = sorted({t["server"] for t in tools.values()})
    sessions = {}

    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            for s in servers:
                params = StdioServerParameters(command=sys.executable, args=["-m", "emerkit.runtime.servers", catalogue_path, s, sink_dir])
                r, w = await stack.enter_async_context(stdio_client(params))
                sess = await stack.enter_async_context(ClientSession(r, w)); await sess.initialize()
                sessions[s] = sess
            yield

    app = FastMCP("emerkit-gateway", lifespan=lifespan)
    for tid, t in tools.items():
        def make(tid=tid, t=t):
            sig = ", ".join(f"{p}: str = ''" for p in t["params"])
            async def forward(**kw):
                with open(log_path, "a") as f:
                    f.write(json.dumps(dict(tool=tid, server=t["server"], args=kw, t=time.time())) + "\n")
                res = await sessions[t["server"]].call_tool(f"{t['name']}__{tid}", kw)
                return "\n".join(c.text for c in res.content if getattr(c, "type", "") == "text")
            ns = {"_f": forward}
            exec(f"async def tool_fn({sig}):\n    return await _f({', '.join(f'{p}={p}' for p in t['params'])})", ns)
            return ns["tool_fn"]
        app.add_tool(make(), name=f"{t['name']}__{tid}", description=t["desc"])
    return app


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]).run("stdio")
