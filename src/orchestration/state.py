from operator import add
from typing import Annotated, TypedDict


class TravelState(TypedDict):
    trip_goal: str
    origin: str
    destination: str
    dates: dict
    travellers: int
    budget_total: float
    budget_remaining: float
    budget_allocations: dict
    constraints: list[str]
    preferences: list[str]
    task_queue: list[str]
    selected_flight: dict | None
    selected_hotel: dict | None
    selected_itinerary: dict | None
    estimated_total_cost: float
    estimated_activities_cost: float
    planner_notes: Annotated[list[str], add]
    reasoning_trace: Annotated[list[dict], add]
    replanning_history: Annotated[list[dict], add]
    rejected_options: dict
    booking_status: dict
    approval: str | None
    next_action: str
    traveller: dict


def new_state(request: str, traveller: dict | None = None) -> TravelState:
    return TravelState(
        trip_goal=request,
        origin="",
        destination="",
        dates={},
        travellers=0,
        budget_total=0.0,
        budget_remaining=0.0,
        budget_allocations={},
        constraints=[],
        preferences=[],
        task_queue=[],
        selected_flight=None,
        selected_hotel=None,
        selected_itinerary=None,
        estimated_total_cost=0.0,
        estimated_activities_cost=0.0,
        planner_notes=[],
        reasoning_trace=[],
        replanning_history=[],
        rejected_options={"flights": [], "hotels": []},
        booking_status={},
        approval=None,
        next_action="",
        traveller=traveller or {},
    )


def emit(entry: dict) -> None:
    """Push a single trace entry to the active stream (for live UI updates).

    A no-op when called outside a LangGraph run (e.g. in tests)."""
    try:
        from langgraph.config import get_stream_writer
        writer = get_stream_writer()
    except Exception:
        writer = None
    if writer:
        writer(entry)


def trace(agent: str, thought: str, action: str, observation: str = "") -> dict:
    entry = {
        "agent": agent,
        "thought": thought,
        "action": action,
        "observation": observation,
    }
    emit(entry)
    return {"reasoning_trace": [entry]}
