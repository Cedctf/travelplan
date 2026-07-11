from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv(override=False)

_REASONER_MARKERS = ("reasoner", "r1")


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    deepseek_api_key: str
    model: str
    duffel_api_key: str
    liteapi_api_key: str
    places_api_key: str


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable {name!r}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def _validate_model(model: str) -> None:
    lowered = model.lower()
    if any(marker in lowered for marker in _REASONER_MARKERS):
        raise ConfigError(
            f"MODEL={model!r} looks like a reasoner/R1 model, which does not "
            f"support tool calling. Use a tool-calling chat model such as "
            f"'deepseek-chat'."
        )


def _load_settings() -> Settings:
    model = _require("MODEL")
    _validate_model(model)
    return Settings(
        deepseek_api_key=_require("DEEPSEEK_API_KEY"),
        model=model,
        duffel_api_key=_require("DUFFEL_API_KEY"),
        liteapi_api_key=_require("LITEAPI_API_KEY"),
        places_api_key=_require("PLACES_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _load_settings()
