from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage

async def create_agent():
    """Создание агента с инструментами."""
    tools = await get_all_tools()
    model = ChatDeepSeek(model="deepseek-chat", temperature=0.3)

    system_prompt = """Ты профессиональный помощник по созданию и редактированию писем и документов.

У тебя есть польный доступ к файловой системе через MCP инструменты:
- read_file, write_file - чтение и запись файлов
- list_directory - просмотр содержимого папок
- create_directory - создание папок
- move_file, copy_file - операции с файлами

ПРИНЦИПЫ РАБОТЫ:
1. Помогай пользователю создавать качественные письма и документы
2. Всегда сохраняй результаты работы в файлы
3. Предлагай улучшение и редактирование
4. Поддерживай контекст сессии - помни о созданых файлах и документах

ЗАВЕРШЕНИЕ СЕССИИ:
Используй end_session() когда:
- Пользователь явно просит завершить ("закончить", "выйти", "хватит")
- Работа полностью выполнена и пользователь доволен результатом
- После фраз типа "спасибо", "готово", "всё хорошо"

ВАЖНО: Будь полезным, дружелюбным и профессиональным!"""

    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
        prompt=SystemMessage(content=system_prompt)
    )

    print("Агент инициализирован с персистенной памятью")
    return agent 
