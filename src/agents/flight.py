from functools import lru_cache

from src.agents.base import (build_react_agent, collect_tool_results,
                             extract_selection, run_agent_streaming)
from src.config import get_settings
from src.llm import get_llm
from src.memory import rejected_ids
from src.orchestration.state import trace
from src.prompts import FLIGHT_SYSTEM_PROMPT
from src.providers.registry import get_flight_provider
from src.services.selection import select_flight
from src.tools.flights import compare_flights, search_flights, select_flight as select_flight_tool

_TOOLS = [search_flights, compare_flights, select_flight_tool]


@lru_cache(maxsize=None)
def build_flight_agent(llm=None):
    return build_react_agent(llm or get_llm(), _TOOLS, FLIGHT_SYSTEM_PROMPT)


def _task(state: dict) -> str:
    allocation = state.get("budget_allocations", {}).get("flights")
    constraints = ", ".join(state.get("constraints", [])) or "none"
    rejected = state.get("rejected_options", {}).get("flights", [])
    rejected_note = "; ".join(
        f"{o.get('airline')} at {o.get('price')}" for o in rejected) or "none"
    return (
        f"Trip request: {state.get('trip_goal')}\n"
        f"Origin: {state.get('origin')}\n"
        f"Destination: {state.get('destination')}\n"
        f"Dates: {state.get('dates')}\n"
        f"Travellers: {state.get('travellers')}\n"
        f"Flight budget target (TOTAL for the whole trip, all travellers): {allocation}\n"
        f"Constraints: {constraints}\n"
        f"Already rejected as too expensive (pick a cheaper alternative): {rejected_note}\n"
        f"Search and compare flights, then surface your ranked candidates."
    )


def _candidates(messages: list) -> list[dict]:
    ranked = collect_tool_results(messages, "compare_flights")
    return ranked or collect_tool_results(messages, "search_flights")


def _confirm(offer: dict) -> dict:
    try:
        return get_flight_provider().select(offer["id"]) or offer
    except Exception:
        return offer


def flight_node(state: dict) -> dict:
    agent = build_flight_agent()
    messages, reasoning_trace = run_agent_streaming(agent, "flight", _task(state))

    candidates = _candidates(messages)
    cfg = (get_settings().selection or {}).get("flight", {})
    chosen = select_flight(
        candidates,
        state.get("budget_allocations", {}).get("flights"),
        state.get("constraints", []),
        rejected_ids(state, "flights"),
        cfg,
    )
    if chosen is not None:
        selected = _confirm(chosen)
        reasoning_trace = reasoning_trace + [trace(
            "flight",
            f"Selected {selected.get('airline')} at {selected.get('price')} "
            f"deterministically from {len(candidates)} candidate(s).",
            "select_flight", str(selected.get("id")))["reasoning_trace"][0]]
    else:
        selected = extract_selection(messages, "select_flight")

    return {
        "selected_flight": selected,
        "reasoning_trace": reasoning_trace,
    }
