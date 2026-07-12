import ast
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

_MAX_OBSERVATION = 600


def build_react_agent(llm, tools, system_prompt):
    return create_react_agent(llm, tools, prompt=system_prompt)


def run_agent_streaming(agent, agent_name: str, task: str) -> list:
    """Run a ReAct agent, emitting each trace entry to the live stream as soon
    as it is complete (an action once its observation lands, a thought right
    away). Returns the full final message list, exactly like ``agent.invoke``."""
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except Exception:
        writer = None

    messages: list = []
    emitted: set[int] = set()

    def flush(final: bool = False) -> None:
        entries = messages_to_trace(agent_name, messages)
        for i, entry in enumerate(entries):
            if i in emitted:
                continue
            if entry["action"] and not entry["observation"] and not final:
                continue
            if writer:
                writer(entry)
            emitted.add(i)

    for chunk in agent.stream({"messages": [HumanMessage(content=task)]},
                              stream_mode="values"):
        messages = chunk["messages"]
        flush()
    flush(final=True)
    return messages


def messages_to_trace(agent: str, messages: list) -> list[dict]:
    entries = []
    by_call_id = {}
    for message in messages:
        if isinstance(message, AIMessage):
            thought = message.content if isinstance(message.content, str) else ""
            tool_calls = getattr(message, "tool_calls", []) or []
            if not tool_calls and thought.strip():
                entries.append({"agent": agent, "thought": thought,
                                "action": "", "observation": ""})
            for call in tool_calls:
                entry = {
                    "agent": agent,
                    "thought": thought,
                    "action": f"{call['name']}({call['args']})",
                    "observation": "",
                }
                entries.append(entry)
                by_call_id[call["id"]] = entry
                thought = ""
        elif isinstance(message, ToolMessage):
            entry = by_call_id.get(message.tool_call_id)
            if entry is not None:
                entry["observation"] = _truncate(str(message.content))
    return entries


def extract_selection(messages: list, select_tool: str):
    ids = _tool_call_ids(messages, select_tool)
    selection = None
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id in ids:
            selection = _coerce(message.content)
    return selection


def collect_tool_results(messages: list, tool_name: str) -> list:
    ids = _tool_call_ids(messages, tool_name)
    results = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id in ids:
            coerced = _coerce(message.content)
            results.extend(coerced if isinstance(coerced, list) else [coerced])
    return results


def final_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) \
                and message.content.strip():
            return message.content
    return ""


def _tool_call_ids(messages: list, tool_name: str) -> set:
    ids = set()
    for message in messages:
        if isinstance(message, AIMessage):
            for call in (getattr(message, "tool_calls", []) or []):
                if call["name"] == tool_name:
                    ids.add(call["id"])
    return ids


def _truncate(text: str) -> str:
    return text if len(text) <= _MAX_OBSERVATION else text[:_MAX_OBSERVATION] + "..."


def _coerce(content):
    if isinstance(content, (dict, list)):
        return content
    if isinstance(content, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(content)
            except (ValueError, SyntaxError):
                continue
    return content
