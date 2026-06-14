from langchain_core.messages import HumanMessage


async def node_research(state: OrchestratorState):
    msgs = [
        HumanMessage(content=f"Тема исследования: {state['topic']}. "
                             f"Найди 3-5 ссылок и сделай короткую сводку.")
    ]
    res = await researcher.ainvoke({"messages": msgs}, config={"configurable": {"thread_id": "research"}})
    # Извлекаем найденные ссылки из ответа
    text = res["messages"][-1].content
    urls = [u for u in text.split() if u.startswith("http")][:5]
    return {"messages": res["messages"], "urls": urls}

async def node_data(state: OrchestratorState):
    msgs = [
        HumanMessage(content=(
            "На основе предыщей сводки/фактов сформируй таблицу с колонками "
            "[source, insights] и 3-8 строк. "
            f"Запиши CSV в {state['csv_path']}. "
            f"Далее создай таблицу sales_insights(source TEXT, insight TEXT) в БД {state['db_path']}"
            "и оставь все строки. Верни выборку COUNT(*) для проверки."
        ))
    ]
    res = await data_eng.ainvoke({"messages": msgs}, config={"configurable": {"thread_id": "data"}})
    return {"messages": res["messages"]}

async def node_write(state: OrchestratorState):
    msgs = [
        HumanMessage(content=(
            f"Собери читабельный отчет в Markdown по теме '{state['topic']}'. "
            f"Используй данные из CSV {state['csv_path']} и выборку из SQLite {state['db_path']}"
            f"(сделай запрос COUNT(*) из sales_insights). "
            f"Сохрани отчет в {state['report_path']} через fs_write_text. "
            "Верни короткое резюме и путь к файлу."
        ))
    ]
    res = await writer.ainvoke({"messages": msgs}, config={"configurable": {"thread_id": "write"}})
    return {"messages": res["messages"]}

# СБорка графа
graph = StateGraph(OrcestratorState)
graph.add_node("research", node_research)
graph.add_node("data", node_data)
graph.add_node("write", node_write)

graph.set_entry_point("research")
graph.add_edge("research", "data")
graph.add_edge("data", "write")

app = graph.compile(checkpoiter=MemorySaver())