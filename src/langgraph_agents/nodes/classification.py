from typing import Literal, TypedDict

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

class TaskClassification(BaseModel):
    task_type: Literal["code", "dialog", "local"] = Field(
        description=(
            "Тип задачи: code - программирование, "
            "dialog - общение, local - российские реалии"
        )
    )

    confidence: float = Field(
        description="Уверенность в классификации от 0.0 до 1.0",
        ge=0.0, le=1.0
    )

    reasoning: str = Field(
        description="Краткое объяснение выбора task_type, максимум 100 символов",
        max_length=100,
    )

class MultiModelState(TypedDict):
    user_question: str
    task_type: str
    code_analysis: str
    dialog_response: str
    local_context: str
    final_answer: str
    should_continue: bool # продолжать ли работу


# Настройка классификатора (используюем DeepSeek как быструю модель)
classification_parser = JsonOutputParser(pydantic_object=TaskClassification)
classification_prompt = PromptTemplate(
    template="""Определи задачи пользователя:
CODE - вопросы про программироваение, отладку, код, алгоритмы, технологии
DIALOG - обычные вопросы, просьбы о помощи, обучение, объяснения
LOCAL - вопросы про Россию, российские законы, локальные особенности, госуслуги

Вопрос: {question}

{format_instructions}

Верни ТОЛЬКО JSON!""",
    input_variables=["question"],
    partial_variables={"format_instructions": classification_parser.get_format_instructions()}
)

def classify_task_node(state: MultiModelState) -> dict:
    """Узел классификации задачи - используем DeepSeek"""
    question = state["user_question"]

    try:
        print(f"Классифицирую задачу...")

        classification_chain = classification_prompt | deepseek_model | classification_parser
        result = classification_chain.invoke({"question": question})

        task_type = result["task_type"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]

        print(f" Тип: {task_type} ({confidence:.2f}) - {reasoning}")

        return {"task_type": task_type}

    except Exception as e:
        print(f"Ошибка классификации: {e}")
        return {"task_type": "dialog"} # fallback к диалогу

# ------------------ Отказоустойчивость на уровне архитектуры -----------------------

def fallback_node(state: MultiModelState) -> dict:
    """ Узел-fallback при недоступности основных моделей """
    try:
        # Пробуем запасную модель
        backup_response = backup_model.invoke(state["user_question"])
        return {"final_answer": backup_response.content}
    except Exception as e:
        print(f"Ошибка fallback-модели: {e}")
        return {"final_answer": "Все модели временно недоступны"}