from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_amvera import AmveraLLM
from langchain_gigachat.chat_models import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field

load_dotenv()

# three models initialize
deepseek_model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.1    # low temperature for technical tasks
)

amvera_model = AmveraLLM(
    model="llama70b",
    temperature=0.7  # moderate temperature for dialogs
)

gigachat_model = GigaChat(
    model="GigaChat-2-Max",
    temperature=0.3, # average temperature
    verify_ssl_certs=False
)