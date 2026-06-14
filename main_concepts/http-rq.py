import uuid
import aiohttp
import asyncio

from settings import (
    GIGACHAT_API_URL,
    GIGACHAT_AUTH_URL,
    GIGACHAT_MODEL,
    GIGACHAT_SCOPE,
    GIGACHAT_SECRET,
    GIGACHAT_VERIFY_SSL,
)

SSL_OPTION = None if GIGACHAT_VERIFY_SSL else False


async def get_access_token() -> str:
    """Получить OAuth-токен GigaChat."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            GIGACHAT_AUTH_URL,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "RqUID": str(uuid.uuid4()),
                "Authorization": f"Basic {GIGACHAT_SECRET}",
            },
            data={"scope": GIGACHAT_SCOPE},
            ssl=SSL_OPTION,
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return result["access_token"]


async def ask_gigachat_llm(token: str, model_name: str, messages: list) -> dict:
    """Отправить запрос к GigaChat chat/completions."""
    async with aiohttp.ClientSession() as session:
        async with session.post(
            GIGACHAT_API_URL,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            json={
                "model": model_name,
                "messages": messages,
            },
            ssl=SSL_OPTION,
        ) as response:
            response.raise_for_status()
            return await response.json()


async def main():
    token = await get_access_token()
    messages = [
        {"role": "system", "content": "Ты полезный ассистент"},
        {"role": "user", "content": "Привет, расскажи про основные метрики ML в ритейле"},
    ]
    response = await ask_gigachat_llm(token, GIGACHAT_MODEL, messages)
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
