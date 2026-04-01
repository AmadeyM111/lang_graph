from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage

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

response = llm.invoke("Расскажи о GRPO")

print(f"Содержимое: {response.content[:100]}...")
print(f"ID сообщения: {response.id}")

# Метаданные о генерации
metadata = response.response_metadata
print(f"Использовано токенов: {metadata.get('token_usage', {})}")
print(f"Модель: {metadata.get('model_name')}")
print(f"Причина завершения: {metadata.get('finish_reason')}")

# Информация о токенах для оптимизации
usage = response.usage_metadata
print(f"Входные токены: {usage.get('input_tokens')}")
print(f"Исходящие токены: {usage.get('output_tokens')}")

# ---------------- TECNICAL REALIZATION IN LANG GRAPH -----------------------

def response_filter_node(state):
    """ Узел-фильтр для коррекции ответов """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        # Проверяем и корректируем ответ
        if "извините" in last_message.content.lower():
            # Заменяем на более уверенный ответ
            corrected = AIMessage(
                content=last_message.content.replace("Извините", "Позвольте уточнить")
            )
            # Заменяем последнее сообщение
            new_messages = state["messages"][:-1] + [corrected]
            return {"messages": new_messages}

        return state # Возвращаем без изменений