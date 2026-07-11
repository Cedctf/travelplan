from typing import Annotated

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from src.providers.registry import get_hotel_provider


@tool
def search_hotels(location: str, check_in: str, check_out: str, guests: int = 2,
                  country_code: str | None = None) -> list[dict]:
    """Search available hotels in a city for the given dates and guests."""
    return get_hotel_provider().search(location, check_in, check_out, guests,
                                       country_code=country_code)


@tool
def compare_hotels(rates: list[dict]) -> list[dict]:
    """Rank hotel offers from cheapest to most expensive."""
    return get_hotel_provider().compare(rates)


@tool
def select_hotel(offer_id: str) -> dict:
    """Prebook a chosen hotel offer to confirm its price."""
    return get_hotel_provider().select(offer_id)


@tool
def book_hotel(offer_id: str, guest: dict,
               state: Annotated[dict, InjectedState]) -> dict:
    """Book a selected hotel rate. Requires approved human-in-the-loop consent."""
    if state.get("approval") != "approved":
        raise PermissionError("Hotel booking requires approval == 'approved'.")
    return get_hotel_provider().book(offer_id, guest)
