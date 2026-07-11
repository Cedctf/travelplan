from src.orchestration.state import trace
from src.providers.base import ProviderError
from src.providers.registry import get_flight_provider, get_hotel_provider

_TEST_TRAVELLER = {
    "title": "mr",
    "given_name": "Test",
    "family_name": "Traveller",
    "gender": "m",
    "born_on": "1990-01-01",
    "email": "test@example.com",
    "phone_number": "+447700900000",
    "firstName": "Test",
    "lastName": "Traveller",
}


def _book_flight(state: dict) -> dict:
    flight = state.get("selected_flight")
    if not flight or not flight.get("id"):
        return {"status": "skipped"}
    try:
        return {"status": "confirmed",
                **get_flight_provider().book(flight["id"], [_TEST_TRAVELLER])}
    except ProviderError as exc:
        return {"status": "failed", "error": str(exc)}


def _book_hotel(state: dict) -> dict:
    hotel = state.get("selected_hotel")
    if not hotel or not hotel.get("id"):
        return {"status": "skipped"}
    try:
        return {"status": "confirmed",
                **get_hotel_provider().book(hotel["id"], _TEST_TRAVELLER)}
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
