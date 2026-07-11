from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.llm import get_llm
from src.orchestration.state import trace
from src.tools.budget import CostSummary, analyze_savings, budget_calculator

_COMPONENT_TO_AGENT = {"flights": "flight", "hotel": "hotel", "activities": "itinerary"}
_MAX_REPLANS = 3


class Allocations(BaseModel):
    flights: float = Field(description="Budget reserved for flights")
    hotel: float = Field(description="Budget reserved for the hotel")
    activities: float = Field(description="Budget reserved for activities")


class TripPlan(BaseModel):
    origin: str = Field(description="Departure city or airport")
    destination: str = Field(description="Destination city")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: str = Field(description="End date YYYY-MM-DD")
    nights: int = Field(description="Number of nights")
    travellers: int = Field(description="Number of travellers")
    budget_total: float = Field(description="Total budget amount")
    currency: str = Field(description="Budget currency code")
    constraints: list[str] = Field(description="Hard constraints, e.g. nonstop")
    preferences: list[str] = Field(description="Traveller interests")
    budget_allocations: Allocations = Field(description="Reasoned split of the budget")
    task_queue: list[str] = Field(description="Ordered steps to plan the trip")


_INTAKE_PROMPT = """You are the Planner for a travel-planning agent team.

Extract the structured trip details from the user's request and reason an initial
budget allocation across flights, hotel, and activities.

Rules for the allocation:
- Do NOT use fixed percentages. Base it on the destination, season, trip length,
  and number of travellers.
- The three allocations must sum to at most the total budget.
- Leave a sensible activities reserve.

Respond with ONLY a JSON object with exactly these fields:
{
  "origin": string,
  "destination": string,
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "nights": integer,
  "travellers": integer,
  "budget_total": number,
  "currency": string,
  "constraints": [string],
  "preferences": [string],
  "budget_allocations": {"flights": number, "hotel": number, "activities": number},
  "task_queue": [string]
}"""


def _structured_llm(llm=None):
    return (llm or get_llm()).with_structured_output(TripPlan, method="json_mode")


def planner_intake(state: dict, llm=None) -> dict:
    system = f"{_INTAKE_PROMPT}\n\nToday's date is {date.today().isoformat()}. " \
             f"When the request names a month without a year, choose the next " \
             f"future occurrence of that month."
    plan: TripPlan = _structured_llm(llm).invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["trip_goal"]),
    ])
    allocations = plan.budget_allocations
    allocation_dict = {
        "flights": allocations.flights,
        "hotel": allocations.hotel,
        "activities": allocations.activities,
    }
    note = (
        f"Allocated {plan.currency} flights={allocations.flights}, "
        f"hotel={allocations.hotel}, activities={allocations.activities} "
        f"of {plan.budget_total} total."
    )
    return {
        "origin": plan.origin,
        "destination": plan.destination,
        "dates": {"start": plan.start_date, "end": plan.end_date, "nights": plan.nights},
        "travellers": plan.travellers,
        "budget_total": plan.budget_total,
        "budget_remaining": plan.budget_total,
        "budget_allocations": allocation_dict,
        "constraints": plan.constraints,
        "preferences": plan.preferences,
        "task_queue": plan.task_queue,
        "planner_notes": [note],
        **trace("planner", "Parsed the request and reasoned a budget allocation.",
                "extract_and_allocate", note),
    }


def _cost_summary(state: dict) -> CostSummary:
    flight = (state.get("selected_flight") or {}).get("price") or 0.0
    hotel = (state.get("selected_hotel") or {}).get("price") or 0.0
    activities = state.get("budget_allocations", {}).get("activities", 0.0)
    return CostSummary(flights=float(flight), hotel=float(hotel),
                       activities=float(activities))


def evaluate(state: dict) -> dict:
    summary = _cost_summary(state)
    calc = budget_calculator(state.get("budget_total", 0.0), summary)
    updates = {
        "estimated_total_cost": summary.total,
        "budget_remaining": calc["budget_remaining"],
    }
    if not calc["over_budget"]:
        updates["next_action"] = "approve"
        updates.update(trace("planner",
                             f"Total {summary.total} is within budget.",
                             "evaluate", "within budget -> approval"))
        return updates

    replans = len(state.get("replanning_history", []))
    if replans >= _MAX_REPLANS:
        note = (
            f"Still over budget after {replans} replanning attempts; no feasible "
            f"solution found. Propose relaxing a constraint or raising the budget."
        )
        updates["next_action"] = "infeasible"
        updates["planner_notes"] = [note]
        updates.update(trace("planner", note, "evaluate", "infeasible"))
        return updates

    ranked = analyze_savings(summary, state.get("budget_allocations", {}))
    target = ranked[0]["component"]
    updates["next_action"] = _COMPONENT_TO_AGENT[target]
    decision = (
        f"Over budget by {-calc['budget_remaining']}. "
        f"{target} has the most savings headroom; re-dispatch it."
    )
    updates["replanning_history"] = [{
        "estimated_total_cost": summary.total,
        "over_by": -calc["budget_remaining"],
        "target": target,
        "ranked": ranked,
    }]
    updates.update(trace("planner", decision, "analyze_savings", f"target={target}"))
    return updates
