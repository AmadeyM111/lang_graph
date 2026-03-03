import os
from langchain_openai import ChatOpenAI

os.environ["DASHSCOPE_API_KEY"] = "sk-..."

llm = init_chat_model("qwenqwen3.5-flash")

