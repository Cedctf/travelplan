import ast
import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

_MAX_OBSERVATION = 600


def build_react_agent(llm, tools, system_prompt):
    return create_react_agent(llm, tools, prompt=system_prompt)


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
