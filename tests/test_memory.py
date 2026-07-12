from src.memory import SearchMemory, add_rejected, rejected_ids


def test_cache_hit_and_miss():
    mem = SearchMemory()
    assert mem.get("places", {"query": "ramen", "location": "Tokyo"}) is None
    mem.put("places", {"query": "ramen", "location": "Tokyo"}, [{"name": "Ichiran"}])
    assert mem.get("places", {"query": "ramen", "location": "Tokyo"}) == [{"name": "Ichiran"}]


def test_cache_key_is_order_independent():
    mem = SearchMemory()
    mem.put("flights", {"origin": "LHR", "destination": "JFK"}, ["a"])
    assert mem.get("flights", {"destination": "JFK", "origin": "LHR"}) == ["a"]


def test_cache_distinguishes_params():
    mem = SearchMemory()
    mem.put("flights", {"origin": "LHR", "destination": "JFK"}, ["a"])
    assert mem.get("flights", {"origin": "LHR", "destination": "NRT"}) is None


def test_rejected_ids():
    state = {"rejected_options": {"hotels": [{"id": "h1"}, {"id": "h2"}], "flights": []}}
    assert rejected_ids(state, "hotels") == {"h1", "h2"}
    assert rejected_ids(state, "flights") == set()


def test_add_rejected_appends_without_duplicates():
    rejected = {"flights": [], "hotels": []}
    option = {"id": "h1", "price": 2300}
    rejected = add_rejected(rejected, "hotels", option)
    rejected = add_rejected(rejected, "hotels", option)
    assert rejected["hotels"] == [option]
