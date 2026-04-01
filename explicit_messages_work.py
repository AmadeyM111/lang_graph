from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_SECRET = os.getenv("GIGACHAT_SECRET")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max-Preview")


llm = GigaChat(model="GigaChat-2-Max-Preview")
credentials=GIGACHAT_SECRET

def chat_with_context():
    # Инициализация диалога с системным сообщением
    messages = [
        SystemMessage(content="Ты дружелюбный помощник программист. Запоминай информацию о пользователе.")
    ]

    # Первое сообщение пользователя
    user_input_1 = "Привет! Я Амадей, Я изучаю LangGraph"
    messages.append(HumanMessage(content=user_input_1))

    response_1 = llm.invoke(messages)
    messages.append(HumanMessage(content=user_input_1))
    print(f"ИИ: {response_1.content}")

    # Второе сообщение - проверяем память 
    user_input_2 = "Как меня зовут и что я изучаю?"
    messages.append(HumanMessage(content=user_input_2))

    response_2 = llm.invoke(messages)
    messages.append(response_2)
    print(f"ИИ: {response_2.content}")

    # Третье сообщение - продолжение темы
    user_input_3 = "Посоветуй мне книгу, курс или статью на эту тему"
    messages.append(HumanMessages(content=user_input_3))

    response_3 = llm.invoke(messages)
    print(f"ИИ: {response_3.content}")

    print(f"\nОбщее количество сообщений в истории: {len(messages)}")
    return messages
    
# Запуск диалога
history = chat_with_context()

     