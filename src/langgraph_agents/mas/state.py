from langgraph.graph import add_messages


class OrchestratorState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    topic: str              # Тема исследования
    urls: List[str]         # Найденные ссылки
    db_path: str            # Путь к SQLite базе
    csv_path: str           # Путь к CSV
    report_path: str        # Путь к итоговому отчету