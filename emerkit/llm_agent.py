"""LLM agent that enacts a task through the MCP gateway (series F1, policy=llm). Run in Claude Code.
Tool calling uses the provider's native format (Anthropic messages API or OpenAI-compatible chat completions)."""
import json, os
from .llm import LLMClient

SYSTEM = ("You are a learning assistant enacting one task of a learning path for a learner. Use the available tools when they help. "
          "Learner context is given below. Stop when the task is done.")


async def run_agent(session, task, context, max_steps=8, client=None):
    client = client or LLMClient()
    listed = (await session.list_tools()).tools
    tools = [{"name": t.name, "description": t.description or "", "input_schema": t.inputSchema} for t in listed]
    user = f"Task: {task['type']} on {', '.join(task['concepts'])}.\nLearner context:\n" + "\n".join(context)
    if client.provider == "anthropic":
        msgs = [{"role": "user", "content": user}]
        for _ in range(max_steps):
            r = client.c.messages.create(model=client.model, max_tokens=4000, system=SYSTEM, tools=tools, messages=msgs)
            msgs.append({"role": "assistant", "content": r.content})
            calls = [b for b in r.content if b.type == "tool_use"]
            if not calls:
                break
            results = []
            for c in calls:
                res = await session.call_tool(c.name, c.input)
                txt = "\n".join(x.text for x in res.content if getattr(x, "type", "") == "text")
                results.append({"type": "tool_result", "tool_use_id": c.id, "content": txt})
            msgs.append({"role": "user", "content": results})
    else:
        oa_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in tools]
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        for _ in range(max_steps):
            r = client.c.chat.completions.create(model=client.model, messages=msgs, tools=oa_tools)
            m = r.choices[0].message; msgs.append(m)
            if m.content and "<think>" in m.content:
                import re as _re; m.content = _re.sub(r"<think>.*?</think>", "", m.content, flags=_re.S)
            if not m.tool_calls:
                break
            for c in m.tool_calls:
                res = await session.call_tool(c.function.name, json.loads(c.function.arguments or "{}"))
                txt = "\n".join(x.text for x in res.content if getattr(x, "type", "") == "text")
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": txt})
    return msgs
