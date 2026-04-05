import json
from pydantic import BaseModel, Field
from typing import List, Literal, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from config import init_llm


# ---------------- Инициализируем LLM -----------------

llm = init_llm()


# ---------------- Модели данных -----------------

class MessageClassification(BaseModel):
    message_type: Literal["review", "question"] = Field(
        description="Тип сообщения: отзыв или вопрос"
    )
    confidence: float = Field(
        description="Уверенность в классификации от 0.0 до 1.0",
        ge=0.0, le=1.0
    )


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


# ---------------- Создаём парсеры локально -----------------

parser = JsonOutputParser(pydantic_object=ReviewAnalysis)
classification_parser = JsonOutputParser(pydantic_object=MessageClassification)

print("Что генерирует парсер:")
print(parser.get_format_instructions())


# ---------------- Состояние системы -----------------

class SystemState(TypedDict):
    messages: List[BaseMessage]       # История диалога
    current_user_input: str           # Текущее сообщение пользователя
    message_type: str                 # Результат классификации
    should_continue: bool             # Продолжить работу?
    analysis_results: List[dict]      # Накопленные JSON-результаты анализа


# ---------------- Узел 1: Получение ввода -----------------

def user_input_node(state: SystemState) -> dict:
    """Узел получения пользовательского ввода"""
    user_input = input("\nВы: ").strip()

    # Команды выхода
    if user_input.lower() in ["стоп", "stop", "выход", "exit"]:
        return {"should_continue": False}

    if user_input.lower() in ["стат", "статистика", "results"]:
        analysis_results = state.get("analysis_results", [])
        if analysis_results:
            print(f"\nПроанализировано отзывов: {len(analysis_results)}")
            sentiments = [r["analysis"]["sentiment"] for r in analysis_results]
            pos = sentiments.count("positive")
            neg = sentiments.count("negative")
            neu = sentiments.count("neutral")
            print(f"Положительные: {pos}, Отрицательные: {neg}, Нейтральные: {neu}")
        else:
            print("Пока нет проанализированных отзывов")
        return {"should_continue": True}

    return {
        "current_user_input": user_input,
        "should_continue": True
    }


# ---------------- Узел 2: Классификация -----------------

classification_prompt = PromptTemplate(
    template="""Определи, является ли это сообщение отзывом о товаре/услуге или это обычный вопрос.

ОТЗЫВ - это мнение о товаре, услуге, опыте пользователя, оценке качества.
ВОПРОС - это запрос информации, общение, просьба о помощи.

Сообщение: {user_input}

{format_instructions}

Верни ТОЛЬКО JSON!""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": classification_parser.get_format_instructions()}
)


def classify_message_node(state: SystemState) -> dict:
    """Узел классификации сообщения"""
    user_input = state["current_user_input"]

    try:
        print("Определяю тип сообщения...")

        classification_chain = classification_prompt | llm | classification_parser
        result = classification_chain.invoke({"user_input": user_input})

        message_type = result["message_type"]
        confidence = result["confidence"]

        print(f"Тип: {message_type} (уверенность: {confidence:.2f})")

        return {"message_type": message_type}

    except Exception as e:
        print(f"Ошибка классификации: {e}")
        return {"message_type": "question"}


# ---------------- Узел 3: Анализ отзыва -----------------

review_parser = JsonOutputParser(pydantic_object=ReviewAnalysis)

review_analysis_prompt = PromptTemplate(
    template="""Проанализируй этот отзыв клиента:

Отзыв: {review}

{format_instructions}

Верни ТОЛЬКО JSON без дополнительных комментариев!""",
    input_variables=["review"],
    partial_variables={"format_instructions": review_parser.get_format_instructions()}
)


def analyze_review_node(state: SystemState) -> dict:
    """Узел анализа отзыва"""
    user_input = state["current_user_input"]

    try:
        print("Анализирую отзыв...")

        analysis_chain = review_analysis_prompt | llm | review_parser
        analysis_result = analysis_chain.invoke({"review": user_input})

        full_result = {
            "original_review": user_input,
            "analysis": analysis_result
        }

        analysis_results = state.get("analysis_results", [])
        new_analysis_results = analysis_results + [full_result]

        print("\n" + "=" * 60)
        print("АНАЛИЗ ОТЗЫВА (JSON):")
        print("=" * 60)
        print(json.dumps(full_result, ensure_ascii=False, indent=2))
        print("=" * 60)

        messages = state["messages"]
        new_messages = messages + [
            HumanMessage(content=user_input),
            AIMessage(content=f"Отзыв проанализирован: {analysis_result['sentiment']}")
        ]

        return {
            "messages": new_messages,
            "analysis_results": new_analysis_results
        }

    except Exception as e:
        print(f"Ошибка при анализе отзыва: {e}")
        messages = state["messages"]
        new_messages = messages + [
            HumanMessage(content=user_input),
            AIMessage(content="Извините, произошла ошибка при анализе отзыва.")
        ]
        return {"messages": new_messages}


# ---------------- Узел 4: Ответ на вопрос -----------------

def answer_question_node(state: SystemState) -> dict:
    """Узел ответа на вопрос"""
    user_input = state["current_user_input"]

    try:
        print("Отвечаю на вопрос...")

        messages = state["messages"] + [HumanMessage(content=user_input)]

        response = llm.invoke(messages)
        ai_response = response.content

        print(f"ИИ: {ai_response}")

        new_messages = messages + [AIMessage(content=ai_response)]

        return {"messages": new_messages}

    except Exception as e:
        print(f"Ошибка при ответе: {e}")

        messages = state["messages"] + [
            HumanMessage(content=user_input),
            AIMessage(content="Извините, произошла ошибка при обработке вашего вопроса.")
        ]

        return {"messages": messages}