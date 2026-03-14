import os
import uuid
import httpx
from dotenv import load_dotenv

load_dotenv()

# --- Конфигурация GigaChat ---
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

GIGACHAT_SECRET = os.getenv("GIGACHAT_SECRET")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max")
GIGACHAT_VERIFY_SSL = False  # True + путь к сертификату для продакшена


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


def chat(messages: list[dict], token: str) -> dict:
    """Отправить запрос к GigaChat chat/completions."""
    response = httpx.post(
        GIGACHAT_API_URL,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        json={
            "model": GIGACHAT_MODEL,
            "messages": messages,
        },
        verify=GIGACHAT_VERIFY_SSL,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


# --- Основной вызов ---
token = get_access_token()

messages = [
    {"role": "system", "content": "Ты полезный программист-консультант"},
    {"role": "user", "content": "Как написать цикл в Python?"},
    {"role": "assistant", "content": "Используйте for или while. Пример: for i in range(10):"},
    {"role": "user", "content": "А что такое range?"},
]

data = chat(messages, token)
print(data["choices"][0]["message"]["content"])

def chat_with_context():
    """Диалог с сохранением контекста через GigaChat."""
    token = get_access_token()

    messages = [
        {"role": "system", "content": "Ты дружелюбный помощник-программист. Запоминай информацию о пользователе."}
    ]

    # Первое сообщение пользователя
    user_input_1 = "Привет! Меня зовут Амадей, я изучаю Python"
    messages.append({"role": "user", "content": user_input_1})

    response_1 = chat(messages, token)
    assistant_msg_1 = response_1["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": assistant_msg_1})
    print(f"ИИ: {assistant_msg_1}")

    # Второе сообщение — проверяем память
    user_input_2 = "Как меня зовут и что я изучаю?"
    messages.append({"role": "user", "content": user_input_2})

    response_2 = chat(messages, token)
    assistant_msg_2 = response_2["choices"][0]["message"]["content"]
    print(f"ИИ: {assistant_msg_2}")