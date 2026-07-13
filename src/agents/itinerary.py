from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import (build_react_agent, collect_tool_results,
                             final_text, run_agent_streaming)
from src.llm import get_llm
from src.models import ItineraryPlan
from src.orchestration.state import trace
from src.prompts import ITINERARY_STRUCT_PROMPT, ITINERARY_SYSTEM_PROMPT
from src.services.guardrails import check_activities_budget
from src.tools.places import search_places

_TOOLS = [search_places]


@lru_cache(maxsize=None)
def build_itinerary_agent(llm=None):
    return build_react_agent(llm or get_llm(), _TOOLS, ITINERARY_SYSTEM_PROMPT)


def _activities_budget(state: dict) -> float:
    flight = float((state.get("selected_flight") or {}).get("price") or 0.0)
    hotel = float((state.get("selected_hotel") or {}).get("price") or 0.0)
    remaining = float(state.get("budget_total") or 0.0) - flight - hotel
    if remaining > 0:
        return remaining
    return float((state.get("budget_allocations") or {}).get("activities", 0.0))


def _currency(state: dict) -> str:
    return ((state.get("selected_flight") or {}).get("currency")
            or (state.get("selected_hotel") or {}).get("currency") or "")


def _task(state: dict) -> str:
    preferences = ", ".join(state.get("preferences", [])) or "general sightseeing"
    hotel = state.get("selected_hotel") or {}
    currency = _currency(state)
    return (
        f"Trip request: {state.get('trip_goal')}\n"
        f"Destination: {state.get('destination')}\n"
        f"Dates: {state.get('dates')}\n"
        f"Interests: {preferences}\n"
        f"Hotel: {hotel.get('name')} ({hotel.get('address', 'location unknown')})\n"
        f"Activities budget (money left after flight and hotel): "
        f"{round(_activities_budget(state), 2)} {currency}\n"
        f"Trip currency: {currency or 'the budget currency'}\n"
        f"Build a day-by-day itinerary with a definite estimated cost per activity."
    )


def _structured_activities(plan: str, travellers: int, currency: str,
                           llm=None) -> ItineraryPlan | None:
    if not (plan or "").strip():
        return None
    system = ITINERARY_STRUCT_PROMPT.format(travellers=travellers or 1,
                                            currency=currency or "the trip currency")
    structured = (llm or get_llm("flash")).with_structured_output(
        ItineraryPlan, method="json_mode")
    try:
        return structured.invoke([
            SystemMessage(content=system),
            HumanMessage(content=plan),
        ])
    except Exception:
        return None


def itinerary_node(state: dict, llm=None) -> dict:
    agent = build_itinerary_agent()
    messages, reasoning_trace = run_agent_streaming(agent, "itinerary", _task(state))
    plan = final_text(messages)

    breakdown = _structured_activities(plan, state.get("travellers") or 1,
                                       _currency(state), llm)
    if breakdown is not None:
        activities_cost = float(breakdown.activities_total)
        activities = [a.model_dump() for a in breakdown.activities]
    else:
        activities_cost = float((state.get("budget_allocations") or {})
                                .get("activities", 0.0))
        activities = []
        reasoning_trace = reasoning_trace + [trace(
            "itinerary",
            "Could not extract a structured activities total from the plan; "
            "using the budgeted activities allocation as the estimate.",
            "extract_activities_total", "fallback")["reasoning_trace"][0]]

    guard = check_activities_budget(activities_cost, _activities_budget(state))
    if guard["over_budget"]:
        reasoning_trace = reasoning_trace + [trace(
            "itinerary",
            f"Activities estimate {round(activities_cost, 2)} exceeds the "
            f"{round(guard['budget'], 2)} left for activities by {guard['over_by']}.",
            "check_activities_budget", "over budget")["reasoning_trace"][0]]

    itinerary = {
        "plan": plan,
        "places": collect_tool_results(messages, "search_places"),
        "activities": activities,
    }
    return {
        "selected_itinerary": itinerary,
        "estimated_activities_cost": activities_cost,
        "reasoning_trace": reasoning_trace,
    }
