from operator import add
from typing import Annotated, TypedDict


class TravelState(TypedDict):
    trip_goal: str
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
    planner_notes: Annotated[list[str], add]
    reasoning_trace: Annotated[list[dict], add]
    replanning_history: Annotated[list[dict], add]
    rejected_options: dict
    booking_status: dict
    approval: str | None
    next_action: str


def new_state(request: str) -> TravelState:
    return TravelState(
        trip_goal=request,
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
        planner_notes=[],
        reasoning_trace=[],
        replanning_history=[],
        rejected_options={"flights": [], "hotels": []},
        booking_status={},
        approval=None,
        next_action="",
    )


def trace(agent: str, thought: str, action: str, observation: str = "") -> dict:
    return {
        "reasoning_trace": [
            {
                "agent": agent,
                "thought": thought,
                "action": action,
                "observation": observation,
            }
        ]
    }
