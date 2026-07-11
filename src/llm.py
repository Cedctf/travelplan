from __future__ import annotations

from functools import lru_cache

from langchain_deepseek import ChatDeepSeek

from src.config import get_settings


@lru_cache(maxsize=1)
def get_llm() -> ChatDeepSeek:
    settings = get_settings()
    return ChatDeepSeek(
        model=settings.model,
        api_key=settings.deepseek_api_key,
        temperature=0,
    )
