from langchain_core.messages import HumanMessage

from src.agents.base import build_react_agent, extract_selection, messages_to_trace
from src.llm import get_llm
from src.tools.flights import compare_flights, search_flights, select_flight

_TOOLS = [search_flights, compare_flights, select_flight]

_SYSTEM_PROMPT = """You are the Flight Agent, a domain expert for booking flights.

You reason with the ReAct pattern: think, call a tool, observe, repeat.

Your job:
1. Read the trip request and the planner's flight budget target and constraints.
2. Call search_flights with IATA airport codes and YYYY-MM-DD dates.
3. If results come back, call compare_flights to rank them cheapest first.
4. Apply the constraints (e.g. nonstop, budget target) and pick the best option.
5. Call select_flight with that offer's id to finalize your choice.

Rules:
- Always finish by calling select_flight for exactly one offer.
- If search_flights returns an empty list, reason about an alternative (nearby
  airport or adjusted date) and search again before giving up.
- Never book. Selection only.

Example sequence:
  search_flights(origin='LHR', destination='JFK', depart_date='2026-09-15')
  compare_flights(offers=<result>)
  select_flight(offer_id='off_...')"""


def build_flight_agent(llm=None):
    return build_react_agent(llm or get_llm(), _TOOLS, _SYSTEM_PROMPT)


def _task(state: dict) -> str:
    allocation = state.get("budget_allocations", {}).get("flights")
    constraints = ", ".join(state.get("constraints", [])) or "none"
    return (
        f"Trip request: {state.get('trip_goal')}\n"
        f"Origin: {state.get('origin')}\n"
        f"Destination: {state.get('destination')}\n"
        f"Dates: {state.get('dates')}\n"
        f"Travellers: {state.get('travellers')}\n"
        f"Flight budget target: {allocation}\n"
        f"Constraints: {constraints}\n"
        f"Find and select the best flight."
    )


def flight_node(state: dict) -> dict:
    agent = build_flight_agent()
    result = agent.invoke({"messages": [HumanMessage(content=_task(state))]})
    messages = result["messages"]
    return {
        "selected_flight": extract_selection(messages, "select_flight"),
        "reasoning_trace": messages_to_trace("flight", messages),
    }
