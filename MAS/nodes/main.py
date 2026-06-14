async def main():
    # начальное состояние оркестратора
    init = {
        "messages": [HumanMessage(content="Старт")],
        "topic": "_dyn_: влияние погодных условий на продажи кофе в Нидерландах",
        "urls": [],
        "db_path": "data.sqlite",
        "csv_path": "dataset.csv",
        "report_path": "report.md",
    }
    config = {"configurable": {"thread_id": "orchestrator-demo"}}

    result = await app.ainvoke(init, config=config)

    print("\n==== ФИНАЛЬНЫЕ СООБЩЕНИЯ ====")
    for m in result["messages"][-6:]:
        role = type(m).__name__.replace("Message", "")
        print(f"[{role}] {getattr(m, 'content', '')[:300]}")

    print("\nФайлы в рабочей папке:")
    for p in sorted(WORKDIR.glob("*")):
        print(" -", p.name)