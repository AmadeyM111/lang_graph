from langchain_core.messages import HumanMessage, AIMessage

async def print_session_stats(agent, config):
    """ Вывод статистики сессии."""
    try:
        state = await agent.aget_state(config)
        if state and "messages" in state.values:
            messages = state.values["messages"]
            human_messages = [m for m in messages if isinstance(m, HumanMessage)]
            ai_messages = [m for m in messages is isinstance(m, AIMessage)]

            print(f"Сообщение пользователя: {len(human_messages)}")
            print(f"Ответов агента: {len(ai_messages)}")
            print(f"Всего сообщений в сессии: {len(messages)}")
        else:
            print("Статистика недоступна")
    except Exception as e:
        print(f"Ошибка получения статистики: {e}")

async def run_interactive_session():
    """Запуск интерактивной сессии."""
    print("ИНТЕРАКТИВНЫЙ ПОМОЩНИК ПО СОЗДАНИЮ ПИСЕМ")
    print("Команды: 'выход', 'quit', 'стоп' - для завершения")
    print("Просто опишите что нужно создать или отредактировать")

    agent = await create_agent()
    config = {"configurable": {"thread_id": "document-session"}}

    try:
        while True:
            try:
                user_input = input("\nВаш запрос: ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['выход', 'quit', 'exit', 'стоп', 'stop']:
                    print("\nДо свидания!")
                    break

                print("\nОбрабатываю запрос...")

                user_message = HumanMessage(content=user_input)
                response_printed = False
                session_ended = False

                async for chunk in agent.astream(
                    {"messages": [user_message]},
                    config=config,
                    stream_mode="values"
                ):
                    if "messages" in chunk and chunk["messages"]:
                        last_msg = chunk["messages"][-1]

                        if isinstance(last_msg, AIMessage) and not response_printed:
                            print(f"\n{last_msg.content}")
                            response_printed = True
                        
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            for tool_call in last_msg.tool_calls:
                                if tool_call["name"] == "end_session":
                                    session_ended = True

                if session_ended:
                    print("Агент завершил сессию")
                    break
            
            except KeyboardInterrupt:
                print("\n\nПолучен сигнал прерывания")
                break
            
            except Exception as e:
                print(f"\nОшибка при обработке запроса: {e}")
                continue

    finally:
        print("\n" + "=" * 60)
        print("СТАТИСТИКА СЕССИИ")
        await print_session_stats(agent, config)
        print("Сессия завершена")


# Запуск
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_interactive_session())



