from langchain_core.tools import tool

@tool
async def session_status() -> str:
    """Показывает статус текущей сессии и доступные файлы."""
    return "Сессия активна. Используйте filesystem инструменты для работы с документами."

@tool
async def end_session(reason: str = "Пользователь завершил работу") -> str:
    """Завершает текущую сессию работы с документами."""
    printf(f"\nЗавершение сессии: {reason}")
    return f"Сессия завершена. {reason}"

    
