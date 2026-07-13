import json
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from src.providers.registry import get_flight_provider


@tool(response_format="content_and_artifact")
def search_flights(
    origin: Annotated[str, "Origin airport IATA code, e.g. 'LHR', 'NRT'"],
    destination: Annotated[str, "Destination airport IATA code, e.g. 'JFK'"],
    depart_date: Annotated[str, "Outbound date, format YYYY-MM-DD"],
    return_date: str | None = Field(default=None, description="Return date YYYY-MM-DD, or omit for one-way"),
    travellers: Annotated[int, "Number of adult passengers"] = 1,
) -> tuple[str, list[dict]]:
    """Search bookable flight offers for a route and dates. Returns normalized
    offers; pass them to compare_flights next. An empty list means no offers
    were found for this route."""
    offers = get_flight_provider().search(origin, destination, depart_date,
                                          return_date, travellers)
    return json.dumps(offers, default=str), offers


@tool(response_format="content_and_artifact")
def compare_flights(
    offers: Annotated[list[dict], "The list of offers returned by search_flights"],
) -> tuple[str, list[dict]]:
    """Rank flight offers cheapest first. Input is the output of search_flights.
    The ranked list is what the system selects the final flight from."""
    ranked = get_flight_provider().compare(offers)
    return json.dumps(ranked, default=str), ranked


@tool(response_format="content_and_artifact")
def select_flight(
    offer_id: Annotated[str, "The 'id' field of the chosen offer, e.g. 'off_...'"],
) -> tuple[str, dict]:
    """Retrieve full, current details for one chosen flight offer by its id."""
    offer = get_flight_provider().select(offer_id)
    return json.dumps(offer, default=str), offer


@tool
def book_flight(
    offer_id: Annotated[str, "The 'id' of the offer to book"],
    passengers: Annotated[list[dict], "Passenger detail objects required by the airline"],
    state: Annotated[dict, InjectedState],
) -> dict:
    """Book a selected flight. Only call after the user has approved; it raises
    unless approval has been granted."""
    if state.get("approval") != "approved":
        raise PermissionError("Flight booking requires approval == 'approved'.")
    return get_flight_provider().book(offer_id, passengers)
