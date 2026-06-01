from langchain_core.messages import SystemMessage

def manage_context_length(messages, max_messages=20):
    """ Простая стратегия: сохраняем системное сообщение + последние N сообщений """
    if len(messages) <= max_messages:
        return messages

    # Выделяем системные сообщения
    system_messages = [msg for msg in messages if isinstance(msg, SystemMessage)]
    dialog_messages = [msg for msg in messages if not isinstance(msg, SystemMessage)]

    # Берем последнеие сообщения диалога
    recent_messages = dialog_messages[-(max_messages - len(system_messages)):]

    return system_messages + recent_messages

# Применение при каждом запросе
def smart_invoke(llm, messages):
    managed_messages = manage_context_length(messages)
    return llm.invoke(managed_messages)
