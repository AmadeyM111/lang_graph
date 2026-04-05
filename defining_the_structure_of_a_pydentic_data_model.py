import json
from pydantic import BaseModel, Field
from typing import List, Literal, TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END

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


# ---------------- Парсеры -----------------

parser = JsonOutputParser(pydantic_object=ReviewAnalysis)
classification_parser = JsonOutputParser(pydantic_object=MessageClassification)


# ---------------- Состояние системы -----------------

class SystemState(TypedDict):
    messages: List[BaseMessage]
    current_user_input: str
    message_type: str
    should_continue: bool
    analysis_results: List[dict]


# ---------------- Промпты -----------------

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

review_analysis_prompt = PromptTemplate(
    template="""Проанализируй этот отзыв клиента:

Отзыв: {review}

{format_instructions}

Верни ТОЛЬКО JSON без дополнительных комментариев!""",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


# ---------------- Узлы -----------------

def user_input_node(state: SystemState) -> dict:
    """Узел получения пользовательского ввода"""
    user_input = input("\nВы: ").strip()

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


def analyze_review_node(state: SystemState) -> dict:
    """Узел анализа отзыва"""
    user_input = state["current_user_input"]

    try:
        print("Анализирую отзыв...")
        analysis_chain = review_analysis_prompt | llm | parser
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


# ---------------- Маршрутизация -----------------

def route_after_input(state: SystemState) -> str:
    """Маршрутизация после ввода пользователя"""
    if not state.get("should_continue", True):
        return "end"

    if state.get("current_user_input"):
        return "classify"

    return "get_input"


def route_after_classification(state: SystemState) -> str:
    """Маршрутизация после классификации"""
    message_type = state.get("message_type", "question")

    if message_type == "review":
        return "analyze_review"
    else:
        return "answer_question"


def route_continue(state: SystemState) -> str:
    """Проверка продолжения работы"""
    return "get_input" if state.get("should_continue", True) else "end"


# ---------------- Сборка графа -----------------

graph = StateGraph(SystemState)

graph.add_node("get_input", user_input_node)
graph.add_node("classify", classify_message_node)
graph.add_node("analyze_review", analyze_review_node)
graph.add_node("answer_question", answer_question_node)

graph.add_edge(START, "get_input")

graph.add_conditional_edges(
    "get_input",
    route_after_input,
    {
        "classify": "classify",
        "get_input": "get_input",
        "end": END
    }
)

graph.add_conditional_edges(
    "classify",
    route_after_classification,
    {
        "analyze_review": "analyze_review",
        "answer_question": "answer_question"
    }
)

graph.add_conditional_edges(
    "analyze_review",
    route_continue,
    {
        "get_input": "get_input",
        "end": END
    }
)

graph.add_conditional_edges(
    "answer_question",
    route_continue,
    {
        "get_input": "get_input",
        "end": END
    }
)

app = graph.compile()


# ---------------- Запуск -----------------

if __name__ == "__main__":
    print("Умная система: Анализ отзывов + Чат-бот")
    print("Введите отзыв - получите JSON анализ")
    print("Задайте вопрос - получите ответ")
    print("Команды: 'стат' - статистика, 'выход' - завершить")
    print("-" * 60)

    initial_state = {
        "messages": [
            SystemMessage(content="Ты дружелюбный помощник. Отвечай кратко и по делу на вопросы пользователя")
        ],
        "current_user_input": "",
        "message_type": "",
        "should_continue": True,
        "analysis_results": []
    }

    try:
        final_state = app.invoke(initial_state)
        print("\nРабота завершена!")
        print(f"Всего сообщений: {len(final_state.get('messages', []))}")
        print(f"Проанализировано отзывов: {len(final_state.get('analysis_results', []))}")

    except KeyboardInterrupt:
        print("\n\nРабота прервана (Ctrl+C)")
    except Exception as e:
        print(f"\nОшибка системы: {e}")