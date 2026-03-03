import os
from langchain_openai import ChatOpenAI
import aiohttp
import asyncio

#os.environ["DASHSCOPE_API_KEY"] = "sk-..."

"""
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
"""

async def ask_amvera_llm(token: str, model_name: str, message: list):
    uri = f"https://kong-proxy.yc.amwera.ru/api/v1/models/gpt"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "X-Auth-Token": f"Bearer (token)",
    }
    data = {
        "model" = model_name,
        "messages" = messages
    }
    async with aiohttp.ClientSession() as session:
        async with session.post (url, headers=headers, json=data) as response:
            response.raise_for_status()
            result = await response.json()
            return result
    
