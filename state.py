from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List

from config import init_llm

# ---------------- Определение состояния -----------------

class ChatState(TypedDict):
    messages: List[BaseMessage]
    should_continue: bool


# ---------------- Инициализация LLM -----------------

llm = init_llm()


def user_input_node(state: ChatState) -> dict:
    """Узел для получения ввода пользователя"""
    user_input = input("Вы: ")

    # Проверка команды выхода
    if user_input.lower() in ["выход", "quit", "exit", "пока", "bye"]:
        return {"should_continue": False}

    # Добавляем сообщение пользователя
    new_messages = state["messages"] + [HumanMessage(content=user_input)]
    return {"messages": new_messages, "should_continue": True}


# ----------------- Узел ответа от ИИ --------------------

def llm_response_node(state: ChatState) -> dict:
    """Узел для генерации ответа ИИ"""
    response = llm.invoke(state["messages"])
    msg_content = response.content
    
    print(f"ИИ: {msg_content}")

    new_messages = state["messages"] + [AIMessage(content=msg_content)]
    return {"messages": new_messages}


# ---------------- условная функция продолжения --------------------

def should_continue(state: ChatState) -> str:
    """Условная функция для определения продолжения диалога."""
    return "continue" if state.get("should_continue", True) else "end"


# --------------- создание и компиляция графа --------------------

graph = StateGraph(ChatState)

graph.add_node("user_input", user_input_node)
graph.add_node("llm_response", llm_response_node)

graph.add_edge(START, "user_input")
graph.add_edge("user_input", "llm_response")

graph.add_conditional_edges(
    "llm_response",
    should_continue,
    {
        "continue": "user_input",
        "end": END
    }
)

app = graph.compile()


# -------------- Запуск диалогового агента -------------------------

if __name__ == "__main__":
    print("Добро пожаловать в чат с ИИ!")
    print("Для выхода введите: выход, quit, exit, пока или bye")
    print("-" * 50)

    initial_state = {
        "messages": [
            SystemMessage(
                content="Ты дружелюбный помощник. Отвечай коротко и по делу."
            )
        ],
        "should_continue": True
    }

    try:
        final_state = app.invoke(initial_state)

        print("-" * 50)
        print("Чат завершен. До свидания!")
        print(f"Всего сообщений в диалоге: {len(final_state['messages'])}")

    except KeyboardInterrupt:
        print("\n\nЧат прерван пользователем (Ctrl+C)")
    except Exception as e:
        print(f"\nОшибка в работе чата: {e}")