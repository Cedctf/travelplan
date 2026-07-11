from __future__ import annotations

import importlib

import pytest

_ALL_KEYS = ("DEEPSEEK_API_KEY", "MODEL", "DUFFEL_API_KEY",
             "LITEAPI_API_KEY", "PLACES_API_KEY")

_VALID_ENV = {
    "DEEPSEEK_API_KEY": "sk-test",
    "MODEL": "deepseek-chat",
    "DUFFEL_API_KEY": "duffel_test_x",
    "LITEAPI_API_KEY": "sand_x",
    "PLACES_API_KEY": "places_x",
}


def _fresh_config(monkeypatch, env):
    for key in _ALL_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return importlib.reload(importlib.import_module("src.config"))


def test_valid_config_loads(monkeypatch):
    config = _fresh_config(monkeypatch, _VALID_ENV)
    settings = config.get_settings()
    assert settings.model == "deepseek-chat"
    assert settings.duffel_api_key == "duffel_test_x"


def test_reasoner_model_rejected(monkeypatch):
    config = _fresh_config(monkeypatch, {**_VALID_ENV, "MODEL": "deepseek-reasoner"})
    with pytest.raises(config.ConfigError, match="reasoner"):
        config.get_settings()


def test_missing_llm_key_rejected(monkeypatch):
    env = {k: v for k, v in _VALID_ENV.items() if k != "DEEPSEEK_API_KEY"}
    config = _fresh_config(monkeypatch, env)
    with pytest.raises(config.ConfigError, match="DEEPSEEK_API_KEY"):
        config.get_settings()


def test_missing_sandbox_key_rejected(monkeypatch):
    env = {k: v for k, v in _VALID_ENV.items() if k != "DUFFEL_API_KEY"}
    config = _fresh_config(monkeypatch, env)
    with pytest.raises(config.ConfigError, match="DUFFEL_API_KEY"):
        config.get_settings()
