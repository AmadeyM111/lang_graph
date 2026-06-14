import httpx

from llm_factory import get_access_token
from settings import GIGACHAT_API_URL, GIGACHAT_MODEL, GIGACHAT_VERIFY_SSL


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


def run_basic_example():
    """Базовый пример single-turn общения."""
    token = get_access_token()
    messages = [
        {"role": "system", "content": "Ты полезный программист-консультант"},
        {"role": "user", "content": "Как написать цикл в Python?"},
        {"role": "assistant", "content": "Используйте for или while. Пример: for i in range(10):"},
        {"role": "user", "content": "А что такое range?"},
    ]
    data = chat(messages, token)
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    run_basic_example()
    chat_with_context()
