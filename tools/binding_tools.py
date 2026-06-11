from langchain_mcp_adapters.client import MultiServerMCPClient

async def ger_all_tools():
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