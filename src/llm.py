from __future__ import annotations

from functools import lru_cache

from langchain_deepseek import ChatDeepSeek

from src.config import get_settings


@lru_cache(maxsize=4)
def get_llm(tier: str = "pro") -> ChatDeepSeek:
    settings = get_settings()
    return ChatDeepSeek(
        model=settings.model_for(tier),
        api_key=settings.deepseek_api_key,
        temperature=settings.temperature,
    )
