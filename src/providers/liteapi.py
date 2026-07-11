import httpx

from src.config import get_settings
from src.providers.base import BookingFailed, SearchFailed, Unavailable

_BASE_URL = "https://api.liteapi.travel/v3.0"
_TIMEOUT = 45
_SEARCH_LIMIT = 10


class LiteAPIHotelProvider:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().liteapi_api_key
        self._offer_ids: dict[str, str] = {}
        self._offers: dict[str, dict] = {}

    def _headers(self) -> dict:
        return {
            "X-API-Key": self._api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def search(self, location: str, check_in: str, check_out: str,
               guests: int, country_code: str | None = None) -> list[dict]:
        params = {"cityName": location, "limit": _SEARCH_LIMIT}
        if country_code:
            params["countryCode"] = country_code
        try:
            listing = httpx.get(f"{_BASE_URL}/data/hotels", headers=self._headers(),
                                params=params, timeout=_TIMEOUT)
            listing.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchFailed(str(exc)) from exc

        hotels = listing.json().get("data", [])
        names = {hotel.get("id"): hotel.get("name") for hotel in hotels}
        hotel_ids = [hotel.get("id") for hotel in hotels if hotel.get("id")]
        if not hotel_ids:
            return []

        body = {
            "hotelIds": hotel_ids,
            "occupancies": [{"adults": max(1, guests)}],
            "currency": "USD",
            "guestNationality": "US",
            "checkin": check_in,
            "checkout": check_out,
        }
        try:
            rates = httpx.post(f"{_BASE_URL}/hotels/rates", headers=self._headers(),
                               json=body, timeout=_TIMEOUT)
            rates.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchFailed(str(exc)) from exc

        results = []
        for entry in rates.json().get("data", []):
            offer = self._cheapest_offer(entry, names)
            if offer:
                results.append(offer)
        return results

    def compare(self, rates: list[dict]) -> list[dict]:
        return sorted(rates, key=lambda rate: rate["price"])

    def select(self, rate_id: str) -> dict:
        offer_id = self._offer_ids.get(rate_id, rate_id)
        try:
            response = httpx.post(f"{_BASE_URL}/rates/prebook", headers=self._headers(),
                                  json={"offerId": offer_id, "usePaymentSdk": False},
                                  timeout=_TIMEOUT)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise Unavailable(str(exc)) from exc
        data = response.json().get("data", {})
        details = self._offers.get(rate_id, {})
        return {
            **details,
            "provider": "liteapi",
            "prebook_id": data.get("prebookId"),
            "price": data.get("price", details.get("price")),
            "currency": data.get("currency", details.get("currency")),
        }

    def book(self, rate_id: str, guest: dict) -> dict:
        prebook = self.select(rate_id)
        body = {
            "prebookId": prebook["prebook_id"],
            "holder": guest,
            "guests": [guest],
        }
        try:
            response = httpx.post(f"{_BASE_URL}/rates/book", headers=self._headers(),
                                  json=body, timeout=_TIMEOUT)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BookingFailed(str(exc)) from exc
        data = response.json().get("data", {})
        return {
            "provider": "liteapi",
            "booking_id": data.get("bookingId"),
            "confirmation": data.get("bookingReference") or data.get("supplierBookingId"),
            "status": data.get("status", "confirmed"),
        }

    def _cheapest_offer(self, entry: dict, names: dict) -> dict | None:
        hotel_id = entry.get("hotelId")
        best = None
        best_offer_id = None
        for room in entry.get("roomTypes", []):
            offer_id = room.get("offerId")
            for rate in room.get("rates", []):
                total = rate.get("retailRate", {}).get("total", [])
                if not total:
                    continue
                amount = total[0].get("amount")
                currency = total[0].get("currency")
                if amount is None:
                    continue
                if best is None or amount < best["price"]:
                    best = {
                        "id": hotel_id,
                        "provider": "liteapi",
                        "hotel_id": hotel_id,
                        "name": names.get(hotel_id),
                        "price": float(amount),
                        "currency": currency,
                        "board": rate.get("boardName"),
                    }
                    best_offer_id = offer_id
        if best is not None:
            self._offer_ids[hotel_id] = best_offer_id
            self._offers[hotel_id] = best
        return best
