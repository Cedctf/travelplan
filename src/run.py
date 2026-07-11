import sys

from langgraph.types import Command

from src.orchestration.graph import build_graph
from src.orchestration.state import new_state


def _print_trace(entries: list) -> None:
    for entry in entries:
        print(f"[{entry['agent']}] {entry['thought'][:110]}")
        if entry["action"]:
            print(f"    -> {entry['action'][:110]}")


def main() -> None:
    if len(sys.argv) < 2:
        print('usage: python -m src.run "<travel request>"')
        return
    request = sys.argv[1]
    app = build_graph()
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": 50}

    result = app.invoke(new_state(request), config)
    _print_trace(result.get("reasoning_trace", []))

    if "__interrupt__" not in result:
        print("\nEnded without booking. next_action:", result.get("next_action"))
        for note in result.get("planner_notes", []):
            print(" note:", note)
        return

    payload = result["__interrupt__"][0].value
    summary = payload["summary"]
    print("\n=== APPROVAL REQUIRED ===")
    print("Flight:", summary["flight"])
    print("Hotel:", summary["hotel"])
    print("Estimated total:", summary["estimated_total_cost"],
          "/ budget", summary["budget_total"])
    print("Itinerary:\n", (summary["itinerary"] or "")[:600])

    answer = input(f"\n{payload['question']} [approve/reject]: ").strip().lower()
    decision = "approved" if answer.startswith("a") else "rejected"

    final = app.invoke(Command(resume=decision), config)
    if decision == "approved":
        print("\nBooking status:", final.get("booking_status"))
    else:
        print("\nBooking cancelled.")


if __name__ == "__main__":
    main()
