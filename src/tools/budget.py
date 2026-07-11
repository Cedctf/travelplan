from dataclasses import dataclass

_COMPONENTS = ("flights", "hotel", "activities")


@dataclass(frozen=True)
class CostSummary:
    flights: float = 0.0
    hotel: float = 0.0
    activities: float = 0.0

    @property
    def total(self) -> float:
        return self.flights + self.hotel + self.activities


def budget_calculator(budget_total: float, summary: CostSummary) -> dict:
    remaining = budget_total - summary.total
    return {
        "budget_total": budget_total,
        "estimated_total_cost": summary.total,
        "budget_remaining": remaining,
        "over_budget": remaining < 0,
    }


def analyze_savings(summary: CostSummary, allocations: dict) -> list[dict]:
    ranked = [
        {
            "component": name,
            "actual": getattr(summary, name),
            "allocated": allocations.get(name, 0.0),
            "headroom": getattr(summary, name) - allocations.get(name, 0.0),
        }
        for name in _COMPONENTS
    ]
    ranked.sort(key=lambda item: item["headroom"], reverse=True)
    return ranked
