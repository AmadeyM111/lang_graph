from pydantic import BaseModel, Field
from typing import List, Literal

from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langchain_core.output_parsers import JsonOutputParser

from typing import TypedDict, List
from dotenv import load_dotenv
import os

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

class SentimentalAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Тональность отзыва: положительная, отрицательная или нейтральная"
    )
    confidence: float = Field(
        description="Уверенность в анализе от 0.0 до 1.0",
        ge=0.0, # больше или равно 0
        le=1.0 # меньше или равно 1
    )
    key_topics: List[str] = Field(
        description="Ключевые темы, упомянутые в отзыве",
        max_items=5
    )
    summary: str = Field(
        description="Краткое резюме отзыва в одном предложении",
        max_lenght=200
    )

# Создаем парсер на основе нашей модели
parser = JsonOutputParser(pydantic_object=SentimentalAnalysis)

print("Что генерирует парсер:")
print(parser.get_format_instructions())

# ----------------- Использование PromptTemplate ----------------

from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    template=f"""Проанализируй отзыв: {review}

{format_instructions}

ТОЛЬКО JSON!""",
    input_variables=["review"], # Что должен предоставить пользователь
    partial_variables={         # Что заполняется автомаически
        "format_instructions": parser.get_format_instructions()
    }
)