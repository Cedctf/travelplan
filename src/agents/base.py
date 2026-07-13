from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode, create_react_agent

from src.config import get_settings
from src.orchestration.state import emit
from src.providers.base import ProviderError


def _handle_tool_error(exc: Exception) -> str:
    if isinstance(exc, ProviderError):
        return (
            f"Tool call failed — {type(exc).__name__}: {exc}. This is usually a "
            f"transient provider or network issue. Try the call again, adjust the "
            f"arguments, or continue with the results you already have."
        )
    raise exc


def build_react_agent(llm, tools, system_prompt):
    tool_node = ToolNode(tools, handle_tool_errors=_handle_tool_error)
    return create_react_agent(llm, tool_node, prompt=system_prompt)


def run_agent_streaming(agent, agent_name: str, task: str) -> tuple[list, list[dict]]:
    messages: list = []
    entries: list[dict] = []
    emitted: set[int] = set()

    def flush(final: bool = False) -> None:
        nonlocal entries
        entries = messages_to_trace(agent_name, messages)
        for i, entry in enumerate(entries):
            if i in emitted:
                continue
            if entry["action"] and not entry["observation"] and not final:
                continue
            emit(entry)
            emitted.add(i)

    for chunk in agent.stream({"messages": [HumanMessage(content=task)]},
                              stream_mode="values"):
        messages = chunk["messages"]
        flush()
    flush(final=True)
    return messages, entries


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
            selection = message.artifact
    return selection


def collect_tool_results(messages: list, tool_name: str) -> list:
    ids = _tool_call_ids(messages, tool_name)
    results = []
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id in ids:
            artifact = message.artifact
            if isinstance(artifact, list):
                results.extend(artifact)
            elif artifact is not None:
                results.append(artifact)
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
    limit = get_settings().max_observation_chars
    return text if len(text) <= limit else text[:limit] + "..."
