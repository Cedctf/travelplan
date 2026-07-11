from langchain_core.messages import HumanMessage

from src.agents.base import (build_react_agent, collect_tool_results,
                             final_text, messages_to_trace)
from src.llm import get_llm
from src.tools.places import search_places

_TOOLS = [search_places]

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
- End with a clear day-by-day plan as your final message.

Example sequence:
  search_places(query='anime shops', location='Tokyo, Shibuya')
  search_places(query='ramen restaurants', location='Tokyo, Shibuya')
  ...then write the day-by-day itinerary."""


def build_itinerary_agent(llm=None):
    return build_react_agent(llm or get_llm(), _TOOLS, _SYSTEM_PROMPT)


def _task(state: dict) -> str:
    preferences = ", ".join(state.get("preferences", [])) or "general sightseeing"
    hotel = state.get("selected_hotel") or {}
    return (
        f"Trip request: {state.get('trip_goal')}\n"
        f"Destination: {state.get('destination')}\n"
        f"Dates: {state.get('dates')}\n"
        f"Interests: {preferences}\n"
        f"Hotel: {hotel.get('name')} ({hotel.get('address', 'location unknown')})\n"
        f"Build a day-by-day itinerary."
    )


def itinerary_node(state: dict) -> dict:
    agent = build_itinerary_agent()
    result = agent.invoke({"messages": [HumanMessage(content=_task(state))]})
    messages = result["messages"]
    itinerary = {
        "plan": final_text(messages),
        "places": collect_tool_results(messages, "search_places"),
    }
    return {
        "selected_itinerary": itinerary,
        "reasoning_trace": messages_to_trace("itinerary", messages),
    }
