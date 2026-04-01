from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_API_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
GIGACHAT_SECRET = os.getenv("GIGACHAT_SECRET")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max-Preview")
GIGACHAT_VERIFY_SSL = False  # True + путь к сертификату для продакшена
# Создаем диалог с подставленным ответом

llm = GigaChat(
    model="GigaChat-2-Max-Preview",
    credentials=GIGACHAT_SECRET,
    verify_ssl_certs=GIGACHAT_VERIFY_SSL
    )

messages = [
    SystemMessage(content="Ты дружелюбный помощник программист. Запоминай информацию о пользователе."),
    HumanMessage(content="Что такое state graph в LangGraph?"),
    AIMessage(content="State graph - это граф, который описывает состояния и переходы между ними."),
    HumanMessage(content="Приведи пример использования state graph в LangGraph."),
]

response = llm.invoke(messages)
print(response.content)