import httpx

from src.config import get_settings
from src.providers.base import BookingFailed, SearchFailed, Unavailable

_BASE_URL = "https://api.duffel.com"
_VERSION = "v2"
_TIMEOUT = 30


class DuffelFlightProvider:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().duffel_api_key

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Duffel-Version": _VERSION,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def search(self, origin: str, destination: str, depart_date: str,
               return_date: str | None, travellers: int) -> list[dict]:
        slices = [{"origin": origin, "destination": destination,
                   "departure_date": depart_date}]
        if return_date:
            slices.append({"origin": destination, "destination": origin,
                           "departure_date": return_date})
        passengers = [{"type": "adult"} for _ in range(max(1, travellers))]
        payload = {"data": {"slices": slices, "passengers": passengers,
                            "cabin_class": "economy"}}
        try:
            response = httpx.post(
                f"{_BASE_URL}/air/offer_requests",
                headers=self._headers(),
                params={"return_offers": "true"},
                json=payload,
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchFailed(str(exc)) from exc
        offers = response.json()["data"].get("offers", [])
        return [self._normalize(offer) for offer in offers]

    def compare(self, offers: list[dict]) -> list[dict]:
        return sorted(offers, key=lambda offer: offer["price"])

    def select(self, offer_id: str) -> dict:
        try:
            response = httpx.get(
                f"{_BASE_URL}/air/offers/{offer_id}",
                headers=self._headers(),
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise Unavailable(str(exc)) from exc
        return self._normalize(response.json()["data"])

    def book(self, offer_id: str, passengers: list[dict]) -> dict:
        selected = self.select(offer_id)
        payload = {
            "data": {
                "type": "instant",
                "selected_offers": [offer_id],
                "passengers": passengers,
                "payments": [{
                    "type": "balance",
                    "amount": f"{selected['price']}",
                    "currency": selected["currency"],
                }],
            }
        }
        try:
            response = httpx.post(
                f"{_BASE_URL}/air/orders",
                headers=self._headers(),
                json=payload,
                timeout=_TIMEOUT,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise BookingFailed(str(exc)) from exc
        order = response.json()["data"]
        return {
            "provider": "duffel",
            "order_id": order.get("id"),
            "booking_reference": order.get("booking_reference"),
            "status": "confirmed",
        }

    def _normalize(self, offer: dict) -> dict:
        slices = offer.get("slices", [])
        first = slices[0] if slices else {}
        segments = first.get("segments", [])
        return {
            "id": offer.get("id"),
            "provider": "duffel",
            "origin": first.get("origin", {}).get("iata_code"),
            "destination": first.get("destination", {}).get("iata_code"),
            "airline": offer.get("owner", {}).get("name"),
            "price": float(offer.get("total_amount", 0.0)),
            "currency": offer.get("total_currency"),
            "stops": max(len(segments) - 1, 0),
        }
