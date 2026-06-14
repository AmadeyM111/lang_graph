from langchain_core.messages import AIMessage

from llm_factory import init_llm


def show_response_metadata() -> None:
    """Запрос к модели и печать метаданных ответа."""
    llm = init_llm()
    response = llm.invoke("Расскажи о GRPO")

    print(f"Содержимое: {response.content[:100]}...")
    print(f"ID сообщения: {response.id}")

    # Метаданные о генерации
    metadata = response.response_metadata
    print(f"Использовано токенов: {metadata.get('token_usage', {})}")
    print(f"Модель: {metadata.get('model_name')}")
    print(f"Причина завершения: {metadata.get('finish_reason')}")

    # Информация о токенах для оптимизации
    usage = response.usage_metadata
    print(f"Входные токены: {usage.get('input_tokens')}")
    print(f"Исходящие токены: {usage.get('output_tokens')}")

# ---------------- TECNICAL REALIZATION IN LANG GRAPH -----------------------

def response_filter_node(state):
    """ Узел-фильтр для коррекции ответов """
    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage):
        # Проверяем и корректируем ответ
        if "извините" in last_message.content.lower():
            # Заменяем на более уверенный ответ
            corrected = AIMessage(
                content=last_message.content.replace("Извините", "Позвольте уточнить")
            )
            # Заменяем последнее сообщение
            new_messages = state["messages"][:-1] + [corrected]
            return {"messages": new_messages}

        return state # Возвращаем без изменений


if __name__ == "__main__":
    show_response_metadata()
