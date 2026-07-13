from src.services.selection import select_flight, select_hotel

_FLIGHT_CFG = {"price_weight": 1.0, "stops_penalty": 0.15,
               "require_nonstop_when_constrained": True}


def _flights():
    return [
        {"id": "a", "airline": "X", "price": 500.0, "stops": 1},
        {"id": "b", "airline": "Y", "price": 450.0, "stops": 2},
        {"id": "c", "airline": "Z", "price": 520.0, "stops": 0},
    ]


def test_flight_pick_is_deterministic():
    first = select_flight(_flights(), 600, [], set(), _FLIGHT_CFG)
    again = select_flight(_flights(), 600, [], set(), _FLIGHT_CFG)
    assert first["id"] == again["id"]


def test_flight_cheapest_when_stops_penalty_small():
    chosen = select_flight(_flights(), 600, [], set(), _FLIGHT_CFG)
    assert chosen["id"] == "b"


def test_flight_nonstop_constraint_filters():
    chosen = select_flight(_flights(), 600, ["nonstop"], set(), _FLIGHT_CFG)
    assert chosen["id"] == "c"  # only the 0-stop offer survives the hard filter


def test_flight_rejected_ids_excluded():
    chosen = select_flight(_flights(), 600, [], {"b"}, _FLIGHT_CFG)
    assert chosen["id"] != "b"


def test_flight_empty_pool_returns_none():
    assert select_flight([], 600, [], set(), _FLIGHT_CFG) is None
    assert select_flight(_flights(), 600, [], {"a", "b", "c"}, _FLIGHT_CFG) is None


def test_hotel_cheapest_wins():
    hotels = [
        {"id": "h1", "name": "A", "price": 900.0},
        {"id": "h2", "name": "B", "price": 700.0},
        {"id": "h3", "name": "C", "price": 800.0},
    ]
    chosen = select_hotel(hotels, 1000, set(), {"price_weight": 1.0})
    assert chosen["id"] == "h2"


def test_hotel_rejected_excluded_and_empty_none():
    hotels = [{"id": "h1", "name": "A", "price": 900.0}]
    assert select_hotel(hotels, 1000, {"h1"}, {}) is None
    assert select_hotel([], 1000, set(), {}) is None
