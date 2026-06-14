from langgraph_mcp import MultiServerMCPClient

async def get_all_tools():
    """Получение всех инструментов: MCP + управление сессией."""
    custom_tools = [session_status, end_session]

    try:
        mcp_client = MultiServerMCPClient({
            "filesystem": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "."],
                "transport": "stdio",
            }
        })
        mcp_tools = await mcp_client.get_tools()
        print(f"Подключено {len(mcp_tools)} инструментов filesystem")
        return custom_tools + mcp_tools
    except Exception as e:
        print(f"MCP недоступен, используем базовые инструменты: {e}")
        return custom_tools