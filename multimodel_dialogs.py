from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import os
import uuid
import httpx
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# --- OpenRouter конфигурация ---
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QWEN_MODEL = "qwen/qwen3.6-plus-preview:free"

# --- GigaChat конфигурация ---
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_SECRET = os.getenv("GIGACHAT_SECRET")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max-Preview")
GIGACHAT_VERIFY_SSL = False


def get_gigachat_token() -> str:
    """Получить OAuth-токен GigaChat."""
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


def ask_openrouter(messages: list[dict]) -> str:
    """Отправить запрос к OpenRouter API и вернуть текст ответа."""
    response = requests.post(
        url=OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": QWEN_MODEL,
            "messages": messages,
        }),
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


def ask_gigachat(messages: list[dict], token: str) -> str:
    """Отправить запрос к GigaChat API и вернуть текст ответа."""
    response = httpx.post(
        GIGACHAT_API_URL,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
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
    result = response.json()
    return result["choices"][0]["message"]["content"]


def langchain_messages_to_dict(messages: list) -> list[dict]:
    """Преобразовать LangChain сообщения в dict для API."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


# --- Основной сценарий ---

# Начальные сообщения
messages = [
    SystemMessage(content="Ты дружелюбный помощник программист. Запоминай информацию о пользователе."),
    HumanMessage(content="Объясни главное отличие между LangGraph и LangChain."),
]

# Конвертируем в формат для API
api_messages = langchain_messages_to_dict(messages)

# Получаем ответ от Qwen через OpenRouter
qwen_response_text = ask_openrouter(api_messages)
print(f"Ответ от Qwen:\n{qwen_response_text}\n")

# Добавляем ответ Qwen в историю как от ассистента
messages.append(AIMessage(content=qwen_response_text))
messages.append(HumanMessage(content="Посмотри на этот ответ и объясни, почему он может быть неполным. Приведи пример из практики."))

# Обновляем сообщения для GigaChat
api_messages = langchain_messages_to_dict(messages)

# Получаем токен GigaChat
giga_token = get_gigachat_token()

# Получаем ответ от GigaChat, считая что история диалога уже включает ответ Qwen
giga_response_text = ask_gigachat(api_messages, giga_token)
print(f"Продолжение от Giga:\n{giga_response_text}")


# --- Создание экспертных персон ---

def create_expert_persona(expertise_area: str) -> list:
    """Создаем экспертную персону с начальным контекстом."""
    return [
        SystemMessage(content=f"Ты эксперт в области {expertise_area}. Ты отвечаешь на вопросы о своей области."),
        HumanMessage(content=f"Как ты можешь помочь в области {expertise_area}?"),
        AIMessage(content=f"Я могу помочь в области {expertise_area} с помощью моих знаний и опыта."),
        HumanMessage(content="Какой у тебя подход к обучению?"),
        AIMessage(content="Я использую методы обучения, которые помогают лучше понять материал. Объясняю сложные концепции через практические примеры и аналогии."),
    ]


def improve_response(original_response: AIMessage) -> AIMessage:
    """Улучшаем ответ от ИИ перед добавлением в контекст."""
    if len(original_response.content) < 50:
        return AIMessage(
            content=f"{original_response.content}\n\nПозвольте мне дать более подробное объяснение..."
        )
    return original_response


# --- Пример использования эксперта ---

if __name__ == "__main__":
    # Создаем эксперта по машинному обучению
    ml_expert_context = create_expert_persona("машинное обучение")
    ml_expert_context.append(HumanMessage(content="Объясни мне нейронные сети"))

    # Конвертируем и отправляем запрос к GigaChat
    ml_api_messages = langchain_messages_to_dict(ml_expert_context)
    giga_token = get_gigachat_token()
    response_text = ask_gigachat(ml_api_messages, giga_token)
    print(f"Ответ эксперта по ML:\n{response_text}\n")

    # Контроль качества — улучшаем короткий ответ
    short_response = AIMessage(content="Короткий ответ.")
    improved = improve_response(short_response)
    print(f"Улучшенный ответ:\n{improved.content}")