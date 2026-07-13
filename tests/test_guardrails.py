from src.services.guardrails import activities_total, check_activities_budget


def test_activities_total_scales_by_travellers():
    acts = [{"cost_per_person": 20.0}, {"cost_per_person": 30.0}]
    assert activities_total(acts, travellers=2) == 100.0
    assert activities_total(acts, travellers=1) == 50.0
    assert activities_total([], travellers=3) == 0.0


def test_check_within_budget():
    result = check_activities_budget(estimated_total=80.0, budget=100.0)
    assert result["over_budget"] is False
    assert result["over_by"] == 0.0


def test_check_over_budget_reports_overage():
    result = check_activities_budget(estimated_total=140.0, budget=100.0)
    assert result["over_budget"] is True
    assert result["over_by"] == 40.0
