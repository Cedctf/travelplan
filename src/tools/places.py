from langchain_core.tools import tool

from src.providers.registry import get_places_provider


@tool
def search_places(query: str, location: str) -> list[dict]:
    """Search points of interest, attractions, or restaurants by query and location."""
    return get_places_provider().search(query, location)
