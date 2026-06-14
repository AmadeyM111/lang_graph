from langgraph.prebuilt import create_react_agent

async def main_react():
    # Получаем все инструменты
    tools = await get_all_tools()

    # Создаем ReAct агента (вся логика графа уже встроена!)
    agent = create_react_agent(
        model=ChatDeepSeek(model="deepseek_chat"),
        tools=tools,
        state_modifier="В твоем распоряжении есть инструмент для работы с файловой системой и получения юридических цитат."
    )

    # Запускаем задачу
    result = await agent.ainvoke({
        "messages": [
            HumanMessage(
                content="Найди юридическую цитату и сохрани ее в файле quote_react.txt c подробной информацией об авторе."
            )
        ] 
    })

    print ("=== Реультат ReAct агента ===")
    print (result["message"][-1].content)

if __name__ == "__main__":
    asyncio.run(main_react())

