import os
import uuid
import httpx
from langchain_gigachat import GigaChat
from dotenv import load_dotenv

load_dotenv()

# --- GigaChat конфигурация ---
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_SECRET = os.getenv("GIGACHAT_SECRET")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max-Preview")
GIGACHAT_VERIFY_SSL = False


def get_access_token() -> str:
    """Получить OAuth-токен GigaChat по client credentials."""
    response = httpx.post(
        GIGACHAT_AUTH_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {GIGACHAT_SECRET}",
        },
        data={"scope": GIGACHAT_SCOPE},
        verify=GIGACHAT_VERIFY_SSL,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def init_llm() -> GigaChat:
    """Инициализировать LLM с получением токена."""
    print("Получаем токен GigaChat...")
    try:
        access_token = get_access_token()
        print("Токен получен успешно!")
    except Exception as e:
        print(f"Ошибка получения токена: {e}")
        print("Проверьте переменную GIGACHAT_SECRET в .env файле")
        exit(1)

    return GigaChat(
        model=GIGACHAT_MODEL,
        temperature=0.0,
        verify_ssl_certs=GIGACHAT_VERIFY_SSL,
        access_token=access_token
    )