from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_all_tools():
    """Получение всех инструментов: ваших + MCP"""
    #  Настройка МСР клиента
    mcp_client = MultiServerMCPClient(
        {
            "filesystem": {
                "comand": "npx",
                "args": ["-y","@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
            }
        }
    )

    # Получаем МСР инструменты
    mcp_tools = await mcp_client.get_tools()

    # Объединяем ваши инструменты с МСР инструментами
    return [get_quote] + mcp_tools

    
    # получаем все инструменты (свои + МСР)
    tools = asyncio.run(get_all_tools())
    llm = ChatDeepSeek(model="deepseek-chat").bind_tools(tools)

    async def model_call(state: AgentState) -> AgentState:
        system_prompt = SystemMessage(
            content="В твоем распоряжении есть инструменты для работы с файловой системой и инструмент для получения юридических цитат философов на русском языке."
        )