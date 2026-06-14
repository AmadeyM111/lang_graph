def mk_researcher():
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.2)
    sys = SystemMessage(content=(
        "Ты Исследователь. Твоя задача: найти 3-5 релевантных ссылок, "
        "при необходимости коротко скачать содержимое по 1-2 из них и выдать сжатую сводку. "
        "Всегда используй инструменты web_search и fetch_url при необходимости"
    ))
    tools = [web_search, fetch_url]
    return create_react_agent(model=model, tools=tools, prompt=sys, checkpointer=MemorySearch())