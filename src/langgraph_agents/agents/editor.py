from langchain_core.messages import SystemMessage


def mk_writer():
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
    sys = SystemMessage(content=(
        "Ты Редактор. Получив данные (сводку, CSV, SQL-выборки), создай читабельный отчет в Markdown. "
        "Сохрани результат на диск через fs_write_text и верни краткое резюме."
    ))
    tools = [fs_write_text, csv_read_rows, sqlite_query]
    return create_react_agent(model=model, tools=tools, prompt=sys, checkpointer=MemoryServer())