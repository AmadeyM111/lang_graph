from pickle import TRUE
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
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

class ChatState(TypedDict):
    messages: List[AIMessage]
#    messages: List[BaseMessage]
    should_continue: bool

llm = GigaChat(model="GigaChat-2-Max-Preview")

def user_input_node(state: ChatState) -> dict:
    """ Узел для получения ввода пользователя """
    user_input = input("Вы: ")

    # Проверка команды выхода
    if user_input.lower() in ["выход", "quit", "exit", "пока", "bye"]:
        return {"should_continue": False}

    # Добавляем сообщение пользователя
    new_messages = state["messages"] + [HumanMessage(content=user_input)]
    return {"messages": new_messages, "should_continue": True}


# ----------------- Узел ответа от ИИ --------------------

def llm_response_node(state: ChatState) -> dict:
    """ Узел для генерации ответа ИИ """
    # Получаем ответ от LLM, передавая весь контекст
    response = llm.invoke(state["messages"])
    msg_content = response.content
    
    # Выводим ответ 
    print(f"ИИ: {msg_content}")

    # Добавляем ответ в историю как AIMessage
    new_messages = state["messages"] + [AIMessage(content=msg_content)]
    return {"messages":new_messages}

# ---------------- условная функция продолжения --------------------

def should_continue(state: ChatState) -> str:
    """ Условная функция для определения продолжения диалога. """
    return "continue" if state.get("should_continue", True) else "end"

# --------------- создание и компиляция графа --------------------

# Создаем граф
graph = StateGraph(ChatState)

# добавляем узлы
graph.add_node("user_input", user_input_node)
graph.add_node("llm_response", llm_response_node)

# Создаем ребра 
graph.add_edge(START, "user_input")
graph.add_edge("user_input", "llm_response")

# Условное ребро для проверки продолжения
graph.add_conditional_edges(
    "llm_response",
    should_continue,
    {
        "continue": "user_input", # Возвращаем к вводу пользователя
        "end": END                # Завершаем диалог
    }
)

# Компиляция графа
app = graph.compile()

# -------------- Запуск диалогового агента -------------------------

if __name__ == "__main__":
    print("Добро пожаловать в чат с ИИ!")
    print("Для выхода введите: выход, quit, exit, пока или bye")
    print("-" * 50)

    # Начальное состояние с системным сообщением
    initial_state ={
        "messages": [
            SystemMessage(
                content="Ты дружелюбный помощник. Отвечай коротко и по делу."
            )
        ],
        "should_continue": True
    }

    try:
        # Запуск чата
        final_state = app.invoke(initial_state)

        print("-" * 50)
        print("Чат завершен. До свидания!")
        print(f"Всего сообщений в диалоге: {len(final_state['messages'])}")

    except KeyboardInterrupt:
        print("\n\nЧат прерван пользователем (Ctrl+C)")
    except Exception as e:
        print(f"\nОшибка в работе чата: {e}")


if __name__ == "__main__":
    print("Добро пожаловать в чат с ИИ")