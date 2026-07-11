import httpx

from src.config import get_settings
from src.providers.base import SearchFailed

_URL = "https://places.googleapis.com/v1/places:searchText"
_TIMEOUT = 30
_FIELD_MASK = (
    "places.displayName,places.formattedAddress,"
    "places.location,places.rating,places.types"
)


class GooglePlacesProvider:
    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or get_settings().places_api_key

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self._api_key,
            "X-Goog-FieldMask": _FIELD_MASK,
        }

    def search(self, query: str, location: str) -> list[dict]:
        text_query = f"{query} in {location}" if location else query
        try:
            response = httpx.post(_URL, headers=self._headers(),
                                  json={"textQuery": text_query}, timeout=_TIMEOUT)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearchFailed(str(exc)) from exc
        places = response.json().get("places", [])
        return [self._normalize(place) for place in places]

    def _normalize(self, place: dict) -> dict:
        location = place.get("location", {})
        return {
            "provider": "google_places",
            "name": place.get("displayName", {}).get("text"),
            "address": place.get("formattedAddress"),
            "rating": place.get("rating"),
            "lat": location.get("latitude"),
            "lng": location.get("longitude"),
            "types": place.get("types", []),
        }
