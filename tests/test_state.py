from langgraph.graph import START, END, StateGraph

from src.orchestration.state import TravelState, new_state, trace


def test_new_state_defaults():
    state = new_state("Plan a 5-day trip to Tokyo")
    assert state["trip_goal"] == "Plan a 5-day trip to Tokyo"
    assert state["reasoning_trace"] == []
    assert state["planner_notes"] == []
    assert state["approval"] is None
    assert state["selected_flight"] is None
    assert state["rejected_options"] == {"flights": [], "hotels": []}


def test_trace_returns_update():
    update = trace("planner", "over budget", "AnalyzeSavings", "hotel has headroom")
    assert update == {
        "reasoning_trace": [
            {
                "agent": "planner",
                "thought": "over budget",
                "action": "AnalyzeSavings",
                "observation": "hotel has headroom",
            }
        ]
    }


def test_reducer_appends_across_nodes():
    def node_a(state):
        return trace("planner", "t1", "a1")

    def node_b(state):
        return trace("flight", "t2", "a2")

    graph = StateGraph(TravelState)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    app = graph.compile()

    result = app.invoke(new_state("req"))
    assert [entry["agent"] for entry in result["reasoning_trace"]] == ["planner", "flight"]
