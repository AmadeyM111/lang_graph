from langchain_gigachat import GigaChat
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


# ---------------- Узел с повторными попытками -----------------

def llm_response_node_with_retry(state: ChatState) -> dict:
    """Узел с обработкой ошибок и повторными попытками"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = llm.invoke(state["messages"])
            msg_content = response.content
            print(f"ИИ: {msg_content}")

            new_messages = state["messages"] + [AIMessage(content=msg_content)]
            return {"messages": new_messages}

        except Exception as e:
            if attempt == max_retries - 1:
                error_msg = "Извините, произошла ошибка. Попробуйте еще раз."
                print(f"ИИ: {error_msg}")
                new_messages = state["messages"] + [AIMessage(content=error_msg)]
                return {"messages": new_messages}
            else:
                print(f"Попытка {attempt + 1} неудачная, повторяю...")
                continue


# ---------------- Контроль длины контекста -----------------

def trim_context_if_needed(messages: List[BaseMessage], max_messages: int = 20) -> List[BaseMessage]:
    """Обрезаем контекст, если он становится слишком длинным"""
    if len(messages) <= max_messages:
        return messages
    
    system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
    dialog_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    recent_msgs = dialog_msgs[-(max_messages - len(system_msgs)):]
    return system_msgs + recent_msgs 


def optimized_llm_response_node(state: ChatState) -> dict:
    """Оптимизированный узел с контролем длины контекста"""
    trimmed_messages = trim_context_if_needed(state["messages"])

    response = llm.invoke(trimmed_messages)
    msg_content = response.content
    print(f"ИИ: {msg_content}")

    new_messages = state["messages"] + [AIMessage(content=msg_content)]
    return {"messages": new_messages}


# ---------------- Обработка пустых сообщений -----------------

def user_input_node(state: ChatState) -> dict:
    """Узел для получения ввода пользователя"""
    user_input = input("Вы: ").strip()

    if not user_input:
        print("Пожалуйста, введите сообщение.")
        return state
    
    if user_input.lower() in ["выход", "quit", "exit", "пока", "bye"]:
        return {"should_continue": False}

    new_messages = state["messages"] + [HumanMessage(content=user_input)]
    return {"messages": new_messages, "should_continue": True}


# ---------------- ИИ принимает решение о завершении диалога -----------------

def ai_controlled_continuation_node(state: ChatState) -> dict:
    """ИИ сам решает, нужно ли завершить диалог"""

    decision_messages = state["messages"] + [
        HumanMessage(
            content="Проанализируй диалог. Если пользователь хочет явно завершить беседу "
            "или диалог исчерпан, ответь ТОЛЬКО словом 'ЗАВЕРШИТЬ'. "
            "Иначе продолжи обычный разговор."
        )
    ]

    response = llm.invoke(decision_messages)

    if "ЗАВЕРШИТЬ" in response.content.upper():
        print("ИИ: Было приятно пообщаться! До свидания!")
        return {"should_continue": False}

    else:
        print(f"ИИ: {response.content}")
        new_messages = state["messages"] + [AIMessage(content=response.content)]
        return {"messages": new_messages, "should_continue": True}


# ---------------- Функция продолжения -----------------

def should_continue(state: ChatState) -> str:
    """Условная функция для определения продолжения диалога"""
    return "continue" if state.get("should_continue", True) else "end"


# ---------------- Создание и компиляция графа -----------------

graph = StateGraph(ChatState)

graph.add_node("user_input", user_input_node)
graph.add_node("llm_response", llm_response_node_with_retry)

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


# ---------------- Запуск диалогового агента -----------------

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