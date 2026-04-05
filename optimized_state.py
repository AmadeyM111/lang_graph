import os
import uuid
import httpx
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from dotenv import load_dotenv

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


# ---------------- Определение состояния -----------------

class ChatState(TypedDict):
    messages: List[BaseMessage]
    should_continue: bool


# ---------------- Инициализация LLM с токеном -----------------

print("Получаем токен GigaChat...")
try:
    access_token = get_access_token()
    print("Токен получен успешно!")
except Exception as e:
    print(f"Ошибка получения токена: {e}")
    print("Проверьте переменную GIGACHAT_SECRET в .env файле")
    exit(1)

llm = GigaChat(
    model=GIGACHAT_MODEL,
    verify_ssl_certs=GIGACHAT_VERIFY_SSL,
    access_token=access_token
)


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
                # Последняя попытка - возвращаем ошибку пользователю
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
    
    # Сохраняем системные сообщения + последние сообщения диалога
    system_msgs = [msg for msg in messages if isinstance(msg, SystemMessage)]
    dialog_msgs = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    recent_msgs = dialog_msgs[-(max_messages - len(system_msgs)):]
    return system_msgs + recent_msgs 


def optimized_llm_response_node(state: ChatState) -> dict:
    """Оптимизированный узел с контролем длины контекста"""
    # Обрезаем контекст при необходимости
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

    if not user_input:  # Пустое сообщение 
        print("Пожалуйста, введите сообщение.")
        return state
    
    if user_input.lower() in ["выход", "quit", "exit", "пока", "bye"]:
        return {"should_continue": False}

    new_messages = state["messages"] + [HumanMessage(content=user_input)]
    return {"messages": new_messages, "should_continue": True}


# ---------------- ИИ принимает решение о завершении диалога -----------------

def ai_controlled_continuation_node(state: ChatState) -> dict:
    """ИИ сам решает, нужно ли завершить диалог"""

    # Добавляем специальный промпт для принятия решения
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
        # Обычный ответ
        print(f"ИИ: {response.content}")
        new_messages = state["messages"] + [AIMessage(content=response.content)]
        return {"messages": new_messages, "should_continue": True}


# ---------------- Функция продолжения -----------------

def should_continue(state: ChatState) -> str:
    """Условная функция для определения продолжения диалога"""
    return "continue" if state.get("should_continue", True) else "end"


# ---------------- Создание и компиляция графа -----------------

graph = StateGraph(ChatState)

# Добавляем узлы
graph.add_node("user_input", user_input_node)
graph.add_node("llm_response", llm_response_node_with_retry)

# Создаем ребра
graph.add_edge(START, "user_input")
graph.add_edge("user_input", "llm_response")

# Условное ребро для проверки продолжения
graph.add_conditional_edges(
    "llm_response",
    should_continue,
    {
        "continue": "user_input",   # Возвращаемся к вводу пользователя
        "end": END                   # Завершаем диалог
    }
)

# Компиляция графа
app = graph.compile()


# ---------------- Запуск диалогового агента -----------------

if __name__ == "__main__":
    print("Добро пожаловать в чат с ИИ!")
    print("Для выхода введите: выход, quit, exit, пока или bye")
    print("-" * 50)

    # Начальное состояние с системным сообщением
    initial_state = {
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