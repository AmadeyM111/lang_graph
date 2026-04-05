import json
from pydantic import BaseModel, Field
from typing import List, Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import init_llm


# ---------------- Определение структуры данных -----------------

class SentimentAnalysis(BaseModel):
    """Модель для анализа тональности отзыва"""
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Тональность отзыва: положительная, отрицательная или нейтральная"
    )
    confidence: float = Field(
        description="Уверенность в анализе от 0.0 до 1.0",
        ge=0.0,
        le=1.0
    )
    key_topics: List[str] = Field(
        description="Ключевые темы, упомянутые в отзыве",
        max_length=5
    )
    summary: str = Field(
        description="Краткое резюме отзыва в одном предложении",
        max_length=200
    )


# ---------------- Создаем парсер -----------------

parser = JsonOutputParser(pydantic_object=SentimentAnalysis)

print("Что генерирует парсер:")
print(parser.get_format_instructions())
print()


# ---------------- Создаем шаблон промпта -----------------

prompt_template = PromptTemplate(
    template="""Проанализируй отзыв: {review}

{format_instructions}

ТОЛЬКО JSON!""",
    input_variables=["review"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


# ---------------- Инициализируем нейросеть -----------------

llm = init_llm()


# ------------------ ТЕСТИРУЕМ МОДЕЛЬ --------------------------

review = "Товар отличный, быстрая доставка! Очень доволен покупкой."

print("==== ПОШАГОВОЕ ВЫПОЛНЕНИЕ ====")
print()

# ШАГ 1: Применяем шаблон
print("ШАГ 1: Применяем PromptTemplate")
prompt_value = prompt_template.invoke({"review": review})
print(f"Тип: {type(prompt_value)}")

# Посмотрим на готовый промпт
prompt_text = prompt_value.to_string()
print("Готовый промпт:")
print(prompt_text[:200] + "...")
print()

# ШАГ 2: Отправляем в нейросеть
print("ШАГ 2: Отправляем в нейросеть")
try:
    llm_response = llm.invoke(prompt_value)
    print(f"Тип ответа: {type(llm_response)}")
    print(f"Ответ: {llm_response.content}")
    print()

    # ШАГ 3: Парсим JSON
    print("ШАГ 3: Парсим JSON")
    parsed_result = parser.invoke(llm_response)
    print(f"Тип результата: {type(parsed_result)}")
    print("Структурированные данные:")
    for key, value in parsed_result.items():
        print(f"    {key}: {value}")

    print()
    print("==== АНАЛИЗ ЗАВЕРШЕН УСПЕШНО ====")
except Exception as e:
    print(f"Ошибка при обращении к GigaChat: {e}")