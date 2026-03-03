import os
from langchain.chat_models import init_chat_model

os.environ["DASHSCOPE_API_KEY"] = "sk-..."

llm = init_chat_model("qwenqwen3.5-flash")
