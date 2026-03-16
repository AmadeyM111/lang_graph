import os
import ssl
import uuid
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_SECRET = os.getenv("GIGACHAT_SECRET")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")

# SSL-контекст без верификации (для продакшена — подключить сертификат)
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


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
            ssl=ssl_context,
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
            ssl=ssl_context,
        ) as response:
            response.raise_for_status()
            return await response.json()


async def main():
    token = await get_access_token()
    messages = [
        {"role": "system", "content": "Ты полезный ассистент"},
        {"role": "user", "content": "Привет, расскажи про основные метрики ML в ритейле"},
    ]
    response = await ask_gigachat_llm(token, "GigaChat-2-Max", messages)
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    asyncio.run(main())
