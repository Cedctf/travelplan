from __future__ import annotations

_STOP_CONSTRAINT_MARKERS = ("nonstop", "non-stop", "non stop", "direct")


def _pool(candidates: list[dict], rejected_ids: set) -> list[dict]:
    return [c for c in (candidates or [])
            if c.get("id") and c["id"] not in (rejected_ids or set())]


def _wants_nonstop(constraints: list[str]) -> bool:
    return any(any(m in (c or "").lower() for m in _STOP_CONSTRAINT_MARKERS)
               for c in (constraints or []))


def _price_norm(pool: list[dict]) -> tuple[float, float]:
    prices = [float(c.get("price") or 0.0) for c in pool]
    lo, hi = min(prices), max(prices)
    return lo, (hi - lo) or 1.0


def select_flight(candidates: list[dict], target: float | None,
                  constraints: list[str] | None, rejected_ids: set | None,
                  cfg: dict | None) -> dict | None:
    cfg = cfg or {}
    pool = _pool(candidates, rejected_ids or set())
    if not pool:
        return None

    if cfg.get("require_nonstop_when_constrained", True) and _wants_nonstop(constraints):
        nonstop = [c for c in pool if int(c.get("stops") or 0) == 0]
        if nonstop:
            pool = nonstop

    lo, span = _price_norm(pool)
    price_weight = float(cfg.get("price_weight", 1.0))
    stops_penalty = float(cfg.get("stops_penalty", 0.15))

    def score(c: dict) -> float:
        norm = (float(c.get("price") or 0.0) - lo) / span
        return price_weight * norm + stops_penalty * int(c.get("stops") or 0)

    return min(pool, key=score)


def select_hotel(candidates: list[dict], target: float | None,
                 rejected_ids: set | None, cfg: dict | None) -> dict | None:
    cfg = cfg or {}
    pool = _pool(candidates, rejected_ids or set())
    if not pool:
        return None

    lo, span = _price_norm(pool)
    price_weight = float(cfg.get("price_weight", 1.0))

    def score(c: dict) -> float:
        return price_weight * ((float(c.get("price") or 0.0) - lo) / span)

    return min(pool, key=score)
