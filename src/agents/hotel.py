from src.agents.base import (build_react_agent, extract_selection,
                             messages_to_trace, run_agent_streaming)
from src.llm import get_llm
from src.tools.hotels import compare_hotels, search_hotels, select_hotel

_TOOLS = [search_hotels, compare_hotels, select_hotel]

_SYSTEM_PROMPT = """You are the Hotel Agent, a domain expert for accommodation.

You reason with the ReAct pattern: think, call a tool, observe, repeat.

Your job:
1. Read the trip request and the planner's hotel budget target.
2. Call search_hotels with the destination city, YYYY-MM-DD dates, guests, and
   the ISO country code when you know it.
3. Call compare_hotels to rank the results cheapest first.
4. Pick the best option within the hotel budget target and near the traveller's
   interests when possible.
5. Call select_hotel with that offer's id to confirm its price.

Rules:
- The hotel budget target and every price you see are TOTALS for the whole
  stay (all nights, all guests), not per-night. Compare the hotel's total price
  directly against the target; do NOT divide the target by the number of nights.
- Always finish by calling select_hotel for exactly one offer.
- If search_hotels returns an empty list, widen the search (nearby area or
  adjusted dates) and try again before giving up.
- Never book. Selection only.

Example sequence:
  search_hotels(location='New York', check_in='2026-12-10', check_out='2026-12-15', guests=2, country_code='US')
  compare_hotels(rates=<result>)
  select_hotel(offer_id='...')"""


def build_hotel_agent(llm=None):
    return build_react_agent(llm or get_llm(), _TOOLS, _SYSTEM_PROMPT)


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
        f"Find and select the best hotel."
    )


def hotel_node(state: dict) -> dict:
    agent = build_hotel_agent()
    messages = run_agent_streaming(agent, "hotel", _task(state))
    return {
        "selected_hotel": extract_selection(messages, "select_hotel"),
        "reasoning_trace": messages_to_trace("hotel", messages),
    }
