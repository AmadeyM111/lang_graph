from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from llm_factory import init_llm


def run_dialog_with_seeded_ai_message() -> None:
    """Демо: продолжаем диалог, где часть истории уже сгенерирована ИИ."""
    llm = init_llm()
    messages = [
        SystemMessage(content="Ты дружелюбный помощник программист. Запоминай информацию о пользователе."),
        HumanMessage(content="Что такое state graph в LangGraph?"),
        AIMessage(content="State graph - это граф, который описывает состояния и переходы между ними."),
        HumanMessage(content="Приведи пример использования state graph в LangGraph."),
    ]
    response = llm.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    run_dialog_with_seeded_ai_message()
