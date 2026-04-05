import os
import uuid
import httpx
from pydantic import BaseModel, Field
from typing import List, Literal

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

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


# ---------------- Получаем токен -----------------

print("Получаем токен GigaChat...")
try:
    access_token = get_access_token()
    print("Токен получен успешно!")
except Exception as e:
    print(f"Ошибка получения токена: {e}")
    print("Проверьте переменную GIGACHAT_SECRET в .env файле")
    exit(1)


# ---------------- Инициализируем LLM -----------------

llm = GigaChat(
    model=GIGACHAT_MODEL,
    temperature=0.0,
    verify_ssl_certs=GIGACHAT_VERIFY_SSL,
    access_token=access_token
)


# ---------------- Модели данных -----------------

# Модель для классификации сообщения
class MessageClassification(BaseModel):
    message_type: Literal["review", "question"] = Field(
        description="Тип сообщения: отзыв или вопрос"
    )
    confidence: float = Field(
        description="Уверенность в классификации от 0.0 до 1.0",
        ge=0.0, le=1.0
    )


# Модель для анализа отзыва
class ReviewAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Тональность отзыва"
    )
    confidence: float = Field(
        description="Уверенность в анализе от 0.0 до 1.0",
        ge=0.0, le=1.0
    )
    key_topics: List[str] = Field(
        description="Ключевые темы из отзыва",
        max_length=5
    )
    summary: str = Field(
        description="Краткое резюме в одном предложении",
        max_length=150
    )


# ---------------- Создаём парсер локально -----------------

parser = JsonOutputParser(pydantic_object=ReviewAnalysis)

print("Что генерирует парсер:")
print(parser.get_format_instructions())