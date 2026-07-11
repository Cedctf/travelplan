from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.providers.registry import get_flight_provider


@tool
def search_flights(origin: str, destination: str, depart_date: str,
                   return_date: str | None = None, travellers: int = 1) -> list[dict]:
    """Search available flights for a route and dates."""
    return get_flight_provider().search(origin, destination, depart_date,
                                        return_date, travellers)


@tool
def compare_flights(offers: list[dict]) -> list[dict]:
    """Rank flight offers from cheapest to most expensive."""
    return get_flight_provider().compare(offers)


@tool
def select_flight(offer_id: str) -> dict:
    """Retrieve full details for a chosen flight offer."""
    return get_flight_provider().select(offer_id)


@tool
def book_flight(offer_id: str, passengers: list[dict],
                state: Annotated[dict, InjectedState]) -> dict:
    """Book a selected flight. Requires approved human-in-the-loop consent."""
    if state.get("approval") != "approved":
        raise PermissionError("Flight booking requires approval == 'approved'.")
    return get_flight_provider().book(offer_id, passengers)
