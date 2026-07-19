from functools import lru_cache

from src.agents.base import (build_react_agent, collect_tool_results,
                             extract_selection, run_agent_streaming)
from src.config import get_settings
from src.llm import get_llm
from src.memory import rejected_ids
from src.orchestration.state import trace
from src.prompts import HOTEL_SYSTEM_PROMPT
from src.providers.registry import get_hotel_provider
from src.services.selection import select_hotel
from src.tools.hotels import compare_hotels, search_hotels, select_hotel as select_hotel_tool

_TOOLS = [search_hotels, compare_hotels, select_hotel_tool]


@lru_cache(maxsize=None)
def build_hotel_agent(llm=None):
    return build_react_agent(llm or get_llm(), _TOOLS, HOTEL_SYSTEM_PROMPT)


def _task(state: dict) -> str:
    allocation = state.get("budget_allocations", {}).get("hotel")
    nights = (state.get("dates") or {}).get("nights")
    preferences = ", ".join(state.get("preferences", [])) or "none"
    rejected = state.get("rejected_options", {}).get("hotels", [])
    rejected_note = "; ".join(
        f"{o.get('name')} at {o.get('price')}" for o in rejected) or "none"
    return (
        f"Trip request: {state.get('trip_goal')}\n"
        f"Destination: {state.get('destination')}\n"
        f"Dates: {state.get('dates')}\n"
        f"Guests: {state.get('travellers')}\n"
        f"Hotel budget target (TOTAL for the whole {nights}-night stay, "
        f"not per night): {allocation}\n"
        f"Traveller interests: {preferences}\n"
        f"Already rejected as too expensive (pick a cheaper alternative): {rejected_note}\n"
        f"Search and compare hotels, then surface your ranked candidates."
    )


def _candidates(messages: list) -> list[dict]:
    ranked = collect_tool_results(messages, "compare_hotels")
    return ranked or collect_tool_results(messages, "search_hotels")


def _confirm(offer: dict) -> dict:
    try:
        return get_hotel_provider().select(offer["id"]) or offer
    except Exception:
        return offer


def hotel_node(state: dict) -> dict:
    agent = build_hotel_agent()
    messages, reasoning_trace = run_agent_streaming(agent, "hotel", _task(state))

    selected = extract_selection(messages, "select_hotel")

    if selected is None:
        candidates = _candidates(messages)
        cfg = (get_settings().selection or {}).get("hotel", {})
        chosen = select_hotel(
            candidates,
            state.get("budget_allocations", {}).get("hotel"),
            rejected_ids(state, "hotels"),
            cfg,
        )
        if chosen is not None:
            selected = _confirm(chosen)
            reasoning_trace = reasoning_trace + [trace(
                "hotel",
                f"LLM did not select; deterministic fallback chose "
                f"{selected.get('name')} at {selected.get('price')} "
                f"from {len(candidates)} candidate(s).",
                "select_hotel_fallback", str(selected.get("id")))["reasoning_trace"][0]]
    else:
        reasoning_trace = reasoning_trace + [trace(
            "hotel",
            f"LLM selected {selected.get('name')} at {selected.get('price')} "
            f"via reasoning.",
            "select_hotel", str(selected.get("id")))["reasoning_trace"][0]]

    return {
        "selected_hotel": selected,
        "reasoning_trace": reasoning_trace,
    }
