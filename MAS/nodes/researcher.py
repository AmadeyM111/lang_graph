def mk_data_engineer():
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.1)
    sys = SystemMessage(content=(
        "Ты Дата-инженер. Тебе дают сводку/факты. "
        "Сформируй таблицу CSV (заголовки + строки), запиши её, "
        "создай/обнови таблицу в SQLite и вставь данные. "
        "Всегда используй csv_write_rows, sqlite_execute, sqlite_query по необходимости. "
    ))
    tools = [csv_write_rows, sqlite_execute, sqlite_query]
    return create_react_agent(model=model, tools=tools, prompt=sys, checkpointer=MemorySaver())