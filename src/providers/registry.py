from functools import lru_cache

from src.providers.duffel import DuffelFlightProvider
from src.providers.liteapi import LiteAPIHotelProvider
from src.providers.places_api import GooglePlacesProvider


@lru_cache(maxsize=1)
def get_flight_provider() -> DuffelFlightProvider:
    return DuffelFlightProvider()


@lru_cache(maxsize=1)
def get_hotel_provider() -> LiteAPIHotelProvider:
    return LiteAPIHotelProvider()


@lru_cache(maxsize=1)
def get_places_provider() -> GooglePlacesProvider:
    return GooglePlacesProvider()
