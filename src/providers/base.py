from typing import Protocol


class ProviderError(Exception):
    pass


class SearchFailed(ProviderError):
    pass


class Unavailable(ProviderError):
    pass


class BookingFailed(ProviderError):
    pass


class FlightProvider(Protocol):
    def search(self, origin: str, destination: str, depart_date: str,
               return_date: str | None, travellers: int) -> list[dict]: ...

    def compare(self, offers: list[dict]) -> list[dict]: ...

    def select(self, offer_id: str) -> dict: ...

    def book(self, offer_id: str, passengers: list[dict]) -> dict: ...


class HotelProvider(Protocol):
    def search(self, location: str, check_in: str, check_out: str,
               guests: int) -> list[dict]: ...

    def compare(self, rates: list[dict]) -> list[dict]: ...

    def select(self, rate_id: str) -> dict: ...

    def book(self, rate_id: str, guest: dict) -> dict: ...


class PlacesProvider(Protocol):
    def search(self, query: str, location: str) -> list[dict]: ...
