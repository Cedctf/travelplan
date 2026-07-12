from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from src.agents.booking import booking_node
from src.agents.flight import flight_node
from src.agents.hotel import hotel_node
from src.agents.itinerary import itinerary_node
from src.agents.planner import budget_gate, evaluate, planner_intake
from src.orchestration.state import TravelState


def human_approval(state: dict) -> dict:
    decision = interrupt({
        "summary": {
            "flight": state.get("selected_flight"),
            "hotel": state.get("selected_hotel"),
            "itinerary": (state.get("selected_itinerary") or {}).get("plan"),
            "estimated_total_cost": state.get("estimated_total_cost"),
            "budget_total": state.get("budget_total"),
        },
        "question": "Would you like me to book this itinerary?",
    })
    return {"approval": decision}


def _route_after_gate(state: dict) -> str:
    if state.get("next_action") == "infeasible":
        return END
    return "itinerary"


def _route_after_evaluate(state: dict) -> str:
    action = state.get("next_action")
    if action == "approve":
        return "human_approval"
    if action == "infeasible":
        return END
    return action


def _route_after_approval(state: dict) -> str:
    return "booking" if state.get("approval") == "approved" else END


def build_graph(checkpointer=None):
    graph = StateGraph(TravelState)
    graph.add_node("planner", planner_intake)
    graph.add_node("flight", flight_node)
    graph.add_node("hotel", hotel_node)
    graph.add_node("budget_gate", budget_gate)
    graph.add_node("itinerary", itinerary_node)
    graph.add_node("evaluate", evaluate)
    graph.add_node("human_approval", human_approval)
    graph.add_node("booking", booking_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "flight")
    graph.add_edge("planner", "hotel")
    graph.add_edge("flight", "budget_gate")
    graph.add_edge("hotel", "budget_gate")
    graph.add_conditional_edges(
        "budget_gate", _route_after_gate,
        ["itinerary", END],
    )
    graph.add_edge("itinerary", "evaluate")
    graph.add_conditional_edges(
        "evaluate", _route_after_evaluate,
        ["human_approval", "flight", "hotel", "itinerary", END],
    )
    graph.add_conditional_edges(
        "human_approval", _route_after_approval, ["booking", END],
    )
    graph.add_edge("booking", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
