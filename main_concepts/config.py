"""Backward compatibility shim.

Основная конфигурация находится в settings.py.
Инициализация LLM находится в llm_factory.py.
"""

from llm_factory import get_access_token, init_gigachat_llm, init_llm
from settings import (
    GIGACHAT_API_URL,
    GIGACHAT_AUTH_URL,
    GIGACHAT_MODEL,
    GIGACHAT_SCOPE,
    GIGACHAT_SECRET,
    GIGACHAT_TEMPERATURE,
    GIGACHAT_VERIFY_SSL,
)

__all__ = [
    "GIGACHAT_AUTH_URL",
    "GIGACHAT_API_URL",
    "GIGACHAT_SECRET",
    "GIGACHAT_SCOPE",
    "GIGACHAT_MODEL",
    "GIGACHAT_TEMPERATURE",
    "GIGACHAT_VERIFY_SSL",
    "get_access_token",
    "init_llm",
    "init_gigachat_llm",
]
