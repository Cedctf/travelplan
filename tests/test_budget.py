from src.tools.budget import CostSummary, analyze_savings, budget_calculator


def test_cost_summary_total():
    summary = CostSummary(flights=2500, hotel=2300, activities=1500)
    assert summary.total == 6300


def test_budget_calculator_over_budget():
    summary = CostSummary(flights=2500, hotel=2300, activities=1500)
    result = budget_calculator(6000, summary)
    assert result["estimated_total_cost"] == 6300
    assert result["budget_remaining"] == -300
    assert result["over_budget"] is True


def test_budget_calculator_within_budget():
    summary = CostSummary(flights=2000, hotel=1800, activities=1200)
    result = budget_calculator(6000, summary)
    assert result["budget_remaining"] == 1000
    assert result["over_budget"] is False


def test_analyze_savings_ranks_highest_headroom_first():
    summary = CostSummary(flights=2500, hotel=2300, activities=1500)
    allocations = {"flights": 2500, "hotel": 2000, "activities": 1500}
    ranked = analyze_savings(summary, allocations)
    assert ranked[0]["component"] == "hotel"
    assert ranked[0]["headroom"] == 300
    assert [item["component"] for item in ranked] == ["hotel", "flights", "activities"]
