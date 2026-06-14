from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import httpx
import requests
import json

from llm_factory import get_access_token
from settings import (
    GIGACHAT_API_URL,
    GIGACHAT_MODEL,
    GIGACHAT_VERIFY_SSL,
    OPENROUTER_API_KEY,
    OPENROUTER_API_URL,
    QWEN_MODEL,
)


def get_gigachat_token() -> str:
    """Получить OAuth-токен GigaChat."""
    return get_access_token()


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


def run_main_scenario() -> None:
    """Запустить основной много-модельный сценарий."""
    messages = [
        SystemMessage(content="Ты дружелюбный помощник программист. Запоминай информацию о пользователе."),
        HumanMessage(content="Объясни главное отличие между LangGraph и LangChain."),
    ]

    api_messages = langchain_messages_to_dict(messages)
    qwen_response_text = ask_openrouter(api_messages)
    print(f"Ответ от Qwen:\n{qwen_response_text}\n")

    messages.append(AIMessage(content=qwen_response_text))
    messages.append(HumanMessage(content="Посмотри на этот ответ и объясни, почему он может быть неполным. Приведи пример из практики."))

    api_messages = langchain_messages_to_dict(messages)
    giga_token = get_gigachat_token()
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
    run_main_scenario()

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
