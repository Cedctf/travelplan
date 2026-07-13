from pydantic import BaseModel, Field


class Allocations(BaseModel):
    flights: float = Field(description="Budget reserved for flights")
    hotel: float = Field(description="Budget reserved for the hotel")
    activities: float = Field(description="Budget reserved for activities")


class TripPlan(BaseModel):
    origin: str = Field(description="Departure city or airport")
    destination: str = Field(description="Destination city")
    start_date: str = Field(description="Start date YYYY-MM-DD")
    end_date: str = Field(description="End date YYYY-MM-DD")
    nights: int = Field(description="Number of nights")
    travellers: int = Field(description="Number of travellers")
    budget_total: float = Field(description="Total budget amount")
    currency: str = Field(description="Budget currency code")
    constraints: list[str] = Field(description="Hard constraints, e.g. nonstop")
    preferences: list[str] = Field(description="Traveller interests")
    budget_allocations: Allocations = Field(description="Reasoned split of the budget")
    task_queue: list[str] = Field(description="Ordered steps to plan the trip")


class ItineraryActivity(BaseModel):
    name: str = Field(description="Activity, attraction, or restaurant name")
    day: int = Field(description="Which day of the trip it falls on, starting at 1")
    cost_per_person: float = Field(
        description="Estimated cost per person in the trip currency; 0 if free")


class ItineraryPlan(BaseModel):
    activities: list[ItineraryActivity] = Field(
        description="Every activity in the itinerary, in order")
    activities_total: float = Field(
        description="Total estimated activities spend for the whole trip and all "
                    "travellers, in the trip currency")
