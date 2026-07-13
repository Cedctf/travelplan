import json
from typing import Annotated

from langchain_core.tools import tool

from src.providers.registry import get_places_provider


@tool(response_format="content_and_artifact")
def search_places(
    query: Annotated[str, "What to look for, e.g. 'anime shops', 'ramen restaurants'"],
    location: Annotated[str, "City or area to search within, e.g. 'Tokyo, Shibuya'"],
) -> tuple[str, list[dict]]:
    """Search points of interest, attractions, or restaurants by query and
    location. Returns places with name, address, rating, and coordinates."""
    places = get_places_provider().search(query, location)
    return json.dumps(places, default=str), places
