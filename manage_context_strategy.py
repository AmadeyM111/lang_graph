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

def manage_context_length(messages, max_messages=20):
    """ Простая стратегия: сохраняем системное сообщение + последние N сообщений """
    if len(messages) &lt;= max_messages:
        return messages

    # Выделяем системные сообщения
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    dialog_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    # Берем последнеие сообщения диалога
    recent_messages = dialog_messages[-(max_messages - len(system_messages)):]

    return system_messages + recent_messages

# Применение при каждом запросе
def smart_invoke(llm, messages):
    managed_messages = manage_context_length(messages)
    return llm.invoke(managed_messages)