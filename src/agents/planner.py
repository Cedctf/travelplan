from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import get_settings
from src.llm import get_llm
from src.memory import add_rejected
from src.models import TripPlan
from src.orchestration.state import trace
from src.prompts import PLANNER_INTAKE_PROMPT
from src.services.budget import CostSummary, analyze_savings, budget_calculator

_COMPONENT_TO_AGENT = {"flights": "flight", "hotel": "hotel", "activities": "itinerary"}


def _structured_llm(llm=None):
    return (llm or get_llm("flash")).with_structured_output(TripPlan, method="json_mode")


def planner_intake(state: dict, llm=None) -> dict:
    system = f"{PLANNER_INTAKE_PROMPT}\n\nToday's date is {date.today().isoformat()}. " \
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


_INFEASIBLE_NOTE = (
    "No feasible plan within budget after {n} attempts. "
    "Try increasing the budget or changing the dates."
)


def _cost_summary(state: dict, use_actual_activities: bool = False) -> CostSummary:
    flight = (state.get("selected_flight") or {}).get("price") or 0.0
    hotel = (state.get("selected_hotel") or {}).get("price") or 0.0
    if use_actual_activities and state.get("estimated_activities_cost"):
        activities = state.get("estimated_activities_cost") or 0.0
    else:
        activities = state.get("budget_allocations", {}).get("activities", 0.0)
    return CostSummary(flights=float(flight), hotel=float(hotel),
                       activities=float(activities))


def _replan(state: dict, summary: CostSummary, calc: dict, allowed: set) -> dict:
    ranked = [row for row in analyze_savings(summary, state.get("budget_allocations", {}))
              if row["component"] in allowed]
    target = ranked[0]["component"]
    updates = {"next_action": _COMPONENT_TO_AGENT[target]}

    rejected = state.get("rejected_options") or {"flights": [], "hotels": []}
    if target == "flights" and state.get("selected_flight"):
        rejected = add_rejected(rejected, "flights", state["selected_flight"])
    elif target == "hotel" and state.get("selected_hotel"):
        rejected = add_rejected(rejected, "hotels", state["selected_hotel"])
    updates["rejected_options"] = rejected

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


def budget_gate(state: dict) -> dict:
    flight = float((state.get("selected_flight") or {}).get("price") or 0.0)
    hotel = float((state.get("selected_hotel") or {}).get("price") or 0.0)
    fixed = flight + hotel
    budget = state.get("budget_total", 0.0)
    updates = {
        "estimated_total_cost": fixed,
        "budget_remaining": budget - fixed,
    }
    if fixed > budget:
        note = (
            f"Even the cheapest flight + hotel ({round(fixed, 2)}) exceed the "
            f"budget ({round(budget, 2)}). Increase the budget or change the dates."
        )
        updates["next_action"] = "infeasible"
        updates["planner_notes"] = [note]
        updates.update(trace("planner", note, "budget_gate", "infeasible"))
        return updates

    updates["next_action"] = "itinerary"
    updates.update(trace(
        "planner",
        f"Flight + hotel ({round(fixed, 2)}) fit within budget ({round(budget, 2)}); "
        f"building the itinerary with the remaining {round(budget - fixed, 2)}.",
        "budget_gate", "within budget -> itinerary"))
    return updates


def evaluate(state: dict) -> dict:
    summary = _cost_summary(state, use_actual_activities=True)
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
    if replans >= get_settings().max_replans:
        note = _INFEASIBLE_NOTE.format(n=replans)
        updates["next_action"] = "infeasible"
        updates["planner_notes"] = [note]
        updates.update(trace("planner", note, "evaluate", "infeasible"))
        return updates

    updates.update(_replan(state, summary, calc,
                           allowed={"flights", "hotel", "activities"}))
    return updates
