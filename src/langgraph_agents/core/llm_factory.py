import uuid

import httpx
from langchain_gigachat import GigaChat

import settings


def get_access_token() -> str:
    """Получить OAuth-токен GigaChat по client credentials."""
    if not settings.GIGACHAT_SECRET:
        raise ValueError("GIGACHAT_SECRET is not set in environment variables.")

    response = httpx.post(
        settings.GIGACHAT_AUTH_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {settings.GIGACHAT_SECRET}",
        },
        data={"scope": settings.GIGACHAT_SCOPE},
        verify=settings.GIGACHAT_VERIFY_SSL,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    token = payload.get("access_token")
    if not token:
        raise ValueError("GigaChat token response does not contain access_token.")
    return token


def init_gigachat_llm() -> GigaChat:
    """Инициализировать GigaChat LLM c access token."""
    access_token = get_access_token()
    return GigaChat(
        model=settings.GIGACHAT_MODEL,
        temperature=settings.GIGACHAT_TEMPERATURE,
        verify_ssl_certs=settings.GIGACHAT_VERIFY_SSL,
        access_token=access_token,
    )


def init_deepseek_llm():
    """Инициализировать DeepSeek LLM (ленивый импорт)."""
    try:
        from langchain_deepseek import ChatDeepSeek
    except ModuleNotFoundError as exc:
        raise ImportError(
            "DeepSeek dependency is missing. Install package: pip install langchain-deepseek"
        ) from exc

    return ChatDeepSeek(
        model=settings.DEEPSEEK_MODEL,
        temperature=settings.DEEPSEEK_TEMPERATURE,
    )


def init_amvera_llm():
    """Инициализировать Amvera LLM (ленивый импорт)."""
    try:
        from langchain_amvera import AmveraLLM
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Amvera dependency is missing. Install package: pip install langchain-amvera"
        ) from exc

    return AmveraLLM(
        model=settings.AMVERA_MODEL,
        temperature=settings.AMVERA_TEMPERATURE,
    )


def init_llm() -> GigaChat:
    """Совместимый алиас для текущих скриптов."""
    return init_gigachat_llm()
