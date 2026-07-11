import pytest

from src.tools.flights import book_flight, search_flights
from src.tools.hotels import book_hotel, search_hotels
from src.tools.places import search_places


def test_tools_have_descriptions():
    assert search_flights.description
    assert search_hotels.description
    assert search_places.description


def test_book_flight_requires_approval():
    with pytest.raises(PermissionError):
        book_flight.func(offer_id="off_x", passengers=[], state={"approval": "pending"})


def test_book_hotel_requires_approval():
    with pytest.raises(PermissionError):
        book_hotel.func(offer_id="rate_x", guest={}, state={"approval": None})
