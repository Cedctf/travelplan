from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv(override=False)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"

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
    flash_model: str = ""
    temperature: float = 0.0
    max_replans: int = 3
    max_observation_chars: int = 600
    selection: dict = None  # type: ignore[assignment]

    def model_for(self, tier: str = "pro") -> str:
        if tier == "flash" and self.flash_model:
            return self.flash_model
        return self.model


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


def _load_yaml() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_settings() -> Settings:
    model = _require("MODEL")
    _validate_model(model)

    cfg = _load_yaml()
    llm = cfg.get("llm", {}) or {}
    flash = (llm.get("models", {}) or {}).get("flash") or ""
    if flash:
        _validate_model(flash)

    return Settings(
        deepseek_api_key=_require("DEEPSEEK_API_KEY"),
        model=model,
        duffel_api_key=_require("DUFFEL_API_KEY"),
        liteapi_api_key=_require("LITEAPI_API_KEY"),
        places_api_key=_require("PLACES_API_KEY"),
        flash_model=flash,
        temperature=float(llm.get("temperature", 0)),
        max_replans=int((cfg.get("planner", {}) or {}).get("max_replans", 3)),
        max_observation_chars=int(
            (cfg.get("agents", {}) or {}).get("max_observation_chars", 600)),
        selection=cfg.get("selection", {}) or {},
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return _load_settings()
