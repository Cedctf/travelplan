import httpx

from src.config import get_settings
from src.memory import MEMORY
from src.providers.base import BookingFailed, SearchFailed, Unavailable

_BASE_URL = "https://api.duffel.com"
_VERSION = "v2"
_TIMEOUT = 40

_DEFAULT_TRAVELLER = {
    "title": "mr", "given_name": "Test", "family_name": "Traveller",
    "born_on": "1990-01-01", "gender": "m", "email": "test@example.com",
    "phone_number": "+442080160509",
}


def _error_detail(exc: httpx.HTTPError) -> str:
    """Include Duffel's response body (which names the offending field) in the
    error, instead of only the generic status-code message."""
    response = getattr(exc, "response", None)
    if response is None:
        return str(exc)
    try:
        errors = response.json().get("errors", [])
    except ValueError:
        return f"{exc} :: {response.text[:300]}"
    parts = [
        " ".join(filter(None, [
            err.get("title"),
            err.get("message"),
            (err.get("source") or {}).get("pointer"),
        ]))
        for err in errors
    ]
    detail = "; ".join(p for p in parts if p)
    return f"{exc} :: {detail}" if detail else str(exc)


class DuffelFlightProvider:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().duffel_api_key
        self._search_ctx: dict[str, dict] = {}

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Duffel-Version": _VERSION,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def _offer_request(self, slices: list[dict], travellers: int) -> tuple[list[str], list[dict]]:
        passengers = [{"type": "adult"} for _ in range(max(1, travellers))]
        payload = {"data": {"slices": slices, "passengers": passengers,
                            "cabin_class": "economy"}}
        response = httpx.post(
            f"{_BASE_URL}/air/offer_requests",
            headers=self._headers(),
            params={"return_offers": "true"},
            json=payload,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()["data"]
        passenger_ids = [p["id"] for p in data.get("passengers", [])]
        return passenger_ids, data.get("offers", [])

    def search(self, origin: str, destination: str, depart_date: str,
               return_date: str | None, travellers: int) -> list[dict]:
        slices = [{"origin": origin, "destination": destination,
                   "departure_date": depart_date}]
        if return_date:
            slices.append({"origin": destination, "destination": origin,
                           "departure_date": return_date})
        params = {"origin": origin, "destination": destination,
                  "depart": depart_date, "return": return_date, "travellers": travellers}
        cached = MEMORY.get("flights", params)
        if cached is not None:
            return cached
        try:
            passenger_ids, offers = self._offer_request(slices, travellers)
        except httpx.HTTPError as exc:
            raise SearchFailed(_error_detail(exc)) from exc
        results = []
        for offer in offers:
            normalized = self._normalize(offer)
            self._search_ctx[normalized["id"]] = {
                "slices": slices,
                "travellers": travellers,
                "owner": normalized["airline"],
            }
            results.append(normalized)
        MEMORY.put("flights", params, results)
        return results

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
            raise Unavailable(_error_detail(exc)) from exc
        return self._normalize(response.json()["data"])

    def book(self, offer_id: str, passengers: list[dict]) -> dict:
        ctx = self._search_ctx.get(offer_id)
        if not ctx:
            raise BookingFailed(f"No search context for offer {offer_id}; re-run search.")
        try:
            passenger_ids, offers = self._offer_request(ctx["slices"], ctx["travellers"])
            offer = self._pick_offer(offers, ctx["owner"])
            if offer is None:
                raise BookingFailed("No bookable offer returned on refresh.")
            order = self._create_order(offer, passenger_ids, passengers)
        except httpx.HTTPError as exc:
            raise BookingFailed(_error_detail(exc)) from exc
        return {
            "provider": "duffel",
            "order_id": order.get("id"),
            "booking_reference": order.get("booking_reference"),
            "airline": offer.get("owner", {}).get("name"),
            "price": float(offer.get("total_amount", 0.0)),
            "currency": offer.get("total_currency"),
            "status": "confirmed",
        }

    def _pick_offer(self, offers: list[dict], owner: str) -> dict | None:
        ranked = sorted(offers, key=lambda offer: float(offer.get("total_amount", 1e12)))
        for offer in ranked:
            if offer.get("owner", {}).get("name") == owner:
                return offer
        return ranked[0] if ranked else None

    def _create_order(self, offer: dict, passenger_ids: list[str],
                      passengers: list[dict]) -> dict:
        people = []
        for index, passenger_id in enumerate(passenger_ids):
            details = passengers[index] if index < len(passengers) else (
                passengers[0] if passengers else _DEFAULT_TRAVELLER)
            people.append({
                "id": passenger_id,
                "title": details.get("title", _DEFAULT_TRAVELLER["title"]),
                "given_name": details.get("given_name", _DEFAULT_TRAVELLER["given_name"]),
                "family_name": details.get("family_name", _DEFAULT_TRAVELLER["family_name"]),
                "born_on": details.get("born_on", _DEFAULT_TRAVELLER["born_on"]),
                "gender": details.get("gender", _DEFAULT_TRAVELLER["gender"]),
                "email": details.get("email", _DEFAULT_TRAVELLER["email"]),
                "phone_number": details.get("phone_number", _DEFAULT_TRAVELLER["phone_number"]),
            })
        payload = {"data": {
            "type": "instant",
            "selected_offers": [offer["id"]],
            "passengers": people,
            "payments": [{
                "type": "balance",
                "amount": offer["total_amount"],
                "currency": offer["total_currency"],
            }],
        }}
        response = httpx.post(f"{_BASE_URL}/air/orders", headers=self._headers(),
                              json=payload, timeout=_TIMEOUT)
        response.raise_for_status()
        return response.json()["data"]

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
