import re

from src.agents.base import (build_react_agent, collect_tool_results,
                             final_text, messages_to_trace, run_agent_streaming)
from src.llm import get_llm
from src.tools.places import search_places

_TOOLS = [search_places]

_TOTAL_RE = re.compile(r"ACTIVITIES_TOTAL:\s*\$?\s*([\d,]+(?:\.\d+)?)")

_SYSTEM_PROMPT = """You are the Itinerary Agent, a domain expert for travel planning.

You reason with the ReAct pattern: think, call a tool, observe, repeat.

Your job:
1. Read the destination, trip length, traveller interests, and hotel location.
2. Use search_places to find attractions, activities, and restaurants that match
   the interests, ideally near the hotel.
3. Build a realistic day-by-day itinerary covering each day of the trip.

Rules:
- Call search_places as many times as needed to cover the different interests.
- Group activities sensibly by day and area to minimise travel.
- For every activity, give a rough estimated cost per person as a single
  definite number in the trip currency (e.g. "25", never a range like
  "20-30" and never symbols like "$$"). Use the place's price_level and
  rating as a guide, or your own knowledge of typical prices. Use 0 for
  free activities.
- End each day with a "Day total" that is the sum of that day's activity
  estimates, and finish the whole plan with an "Estimated activities total".
- Keep the estimated activities total within the activities budget provided.
- On the very last line, output a machine-readable total in exactly this
  format (one definite number, no currency symbol, no range):
      ACTIVITIES_TOTAL: <number>
- End with a clear day-by-day plan as your final message.

Example sequence:
  search_places(query='anime shops', location='Tokyo, Shibuya')
  search_places(query='ramen restaurants', location='Tokyo, Shibuya')
  ...then write the day-by-day itinerary with a definite cost per activity,
  ending with the ACTIVITIES_TOTAL line."""


def build_itinerary_agent(llm=None):
    return build_react_agent(llm or get_llm(), _TOOLS, _SYSTEM_PROMPT)


def _activities_budget(state: dict) -> float:
    """Money actually left for activities after the selected flight and hotel."""
    flight = float((state.get("selected_flight") or {}).get("price") or 0.0)
    hotel = float((state.get("selected_hotel") or {}).get("price") or 0.0)
    remaining = float(state.get("budget_total") or 0.0) - flight - hotel
    if remaining > 0:
        return remaining
    return float((state.get("budget_allocations") or {}).get("activities", 0.0))


def _task(state: dict) -> str:
    preferences = ", ".join(state.get("preferences", [])) or "general sightseeing"
    hotel = state.get("selected_hotel") or {}
    currency = ((state.get("selected_flight") or {}).get("currency")
                or (state.get("selected_hotel") or {}).get("currency") or "")
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


def _parse_activities_total(plan: str, fallback: float) -> float:
    match = _TOTAL_RE.search(plan or "")
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return fallback


def itinerary_node(state: dict) -> dict:
    agent = build_itinerary_agent()
    messages = run_agent_streaming(agent, "itinerary", _task(state))
    plan = final_text(messages)
    fallback = float((state.get("budget_allocations") or {}).get("activities", 0.0))
    itinerary = {
        "plan": plan,
        "places": collect_tool_results(messages, "search_places"),
    }
    return {
        "selected_itinerary": itinerary,
        "estimated_activities_cost": _parse_activities_total(plan, fallback),
        "reasoning_trace": messages_to_trace("itinerary", messages),
    }
