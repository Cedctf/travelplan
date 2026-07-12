from threading import Lock


class SearchMemory:
    def __init__(self):
        self._cache: dict = {}
        self._lock = Lock()

    @staticmethod
    def _key(namespace: str, params: dict):
        return (namespace, tuple(sorted((k, str(v)) for k, v in params.items())))

    def get(self, namespace: str, params: dict):
        with self._lock:
            return self._cache.get(self._key(namespace, params))

    def put(self, namespace: str, params: dict, value) -> None:
        with self._lock:
            self._cache[self._key(namespace, params)] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


MEMORY = SearchMemory()


def rejected_ids(state: dict, category: str) -> set:
    options = (state.get("rejected_options") or {}).get(category, [])
    return {option.get("id") for option in options if option.get("id")}


def add_rejected(rejected_options: dict, category: str, option: dict) -> dict:
    current = dict(rejected_options or {"flights": [], "hotels": []})
    existing = current.get(category, [])
    if option and option not in existing:
        current = {**current, category: existing + [option]}
    return current
