from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from dotenv import load_dotenv
import os

load_dotenv()

class ChatState(TypedDict):
    messages: List[BaseMessage]
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