import os
from langchain_openai import ChatOpenAI
import aiohttp
import asyncio


#os.environ["DASHSCOPE_API_KEY"] = "sk-..."

llm = ChatOpenAI (
    model = "qwenqwen3.5-flash",
    temperature = 0.7,
    max_tokens = 2000,
    timeout = 30,
    max_retries = 3,
    streaming=True
)

response = llm.invoke("Какой диаметр колец Сатурна?")
print(response.content)