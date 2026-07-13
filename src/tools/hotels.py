import json
from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import Field

from src.providers.registry import get_hotel_provider


@tool(response_format="content_and_artifact")
def search_hotels(
    location: Annotated[str, "City name, e.g. 'Tokyo', 'New York'"],
    check_in: Annotated[str, "Check-in date, format YYYY-MM-DD"],
    check_out: Annotated[str, "Check-out date, format YYYY-MM-DD"],
    guests: Annotated[int, "Number of adult guests"] = 2,
    country_code: str | None = Field(default=None, description="ISO 3166 alpha-2 country code, e.g. 'JP'"),
) -> tuple[str, list[dict]]:
    """Search bookable hotel offers in a city for the given dates. Returns one
    cheapest offer per hotel; pass them to compare_hotels next. An empty list
    means no hotels were found."""
    offers = get_hotel_provider().search(location, check_in, check_out, guests,
                                         country_code=country_code)
    return json.dumps(offers, default=str), offers


@tool(response_format="content_and_artifact")
def compare_hotels(
    rates: Annotated[list[dict], "The list of offers returned by search_hotels"],
) -> tuple[str, list[dict]]:
    """Rank hotel offers cheapest first. Input is the output of search_hotels.
    The ranked list is what the system selects the final hotel from."""
    ranked = get_hotel_provider().compare(rates)
    return json.dumps(ranked, default=str), ranked


@tool(response_format="content_and_artifact")
def select_hotel(
    offer_id: Annotated[str, "The 'id' field of the chosen hotel offer"],
) -> tuple[str, dict]:
    """Prebook a chosen hotel offer to confirm its current price before booking."""
    offer = get_hotel_provider().select(offer_id)
    return json.dumps(offer, default=str), offer


@tool
def book_hotel(
    offer_id: Annotated[str, "The 'id' of the offer to book"],
    guest: Annotated[dict, "Lead guest details required by the hotel"],
    state: Annotated[dict, InjectedState],
) -> dict:
    """Book a selected hotel rate. Only call after the user has approved; it
    raises unless approval has been granted."""
    if state.get("approval") != "approved":
        raise PermissionError("Hotel booking requires approval == 'approved'.")
    return get_hotel_provider().book(offer_id, guest)
