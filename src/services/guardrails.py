from __future__ import annotations


def activities_total(activities: list[dict], travellers: int) -> float:
    per_person = sum(float(a.get("cost_per_person") or 0.0) for a in (activities or []))
    return per_person * max(1, int(travellers or 1))


def check_activities_budget(estimated_total: float, budget: float) -> dict:
    over_by = round(max(0.0, float(estimated_total) - float(budget)), 2)
    return {
        "over_budget": over_by > 0,
        "over_by": over_by,
        "budget": float(budget),
    }
