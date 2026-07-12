from src.orchestration.state import trace
from src.providers.base import ProviderError
from src.providers.registry import get_flight_provider, get_hotel_provider

def _traveller(state: dict) -> dict:
    """Traveller details as supplied by the user from the frontend."""
    return {k: v for k, v in (state.get("traveller") or {}).items() if v}


def _book_flight(state: dict) -> dict:
    flight = state.get("selected_flight")
    if not flight or not flight.get("id"):
        return {"status": "skipped"}
    try:
        return {**get_flight_provider().book(flight["id"], [_traveller(state)]),
                "status": "confirmed"}
    except ProviderError as exc:
        return {"status": "failed", "error": str(exc)}


def _book_hotel(state: dict) -> dict:
    hotel = state.get("selected_hotel")
    if not hotel or not hotel.get("id"):
        return {"status": "skipped"}
    try:
        return {**get_hotel_provider().book(hotel["id"], _traveller(state)),
                "status": "confirmed"}
    except ProviderError as exc:
        return {"status": "failed", "error": str(exc)}


def booking_node(state: dict) -> dict:
    if state.get("approval") != "approved":
        return {
            "booking_status": {"overall": "blocked"},
            **trace("planner", "Booking blocked; approval not granted.",
                    "booking", "blocked"),
        }
    flight_result = _book_flight(state)
    hotel_result = _book_hotel(state)
    confirmed = {"confirmed", "skipped"}
    overall = ("confirmed"
               if flight_result["status"] in confirmed
               and hotel_result["status"] in confirmed
               else "failed")
    status = {"flight": flight_result, "hotel": hotel_result, "overall": overall}
    return {
        "booking_status": status,
        **trace("planner", f"Executed booking; overall={overall}.",
                "execute_booking", str(status)[:300]),
    }
