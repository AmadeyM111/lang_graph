from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from llm_factory import init_llm


def chat_with_context():
    llm = init_llm()

    # Инициализация диалога с системным сообщением
    messages = [
        SystemMessage(content="Ты дружелюбный помощник программист. Запоминай информацию о пользователе.")
    ]

    # Первое сообщение пользователя
    user_input_1 = "Привет! Я Амадей, Я изучаю LangGraph"
    messages.append(HumanMessage(content=user_input_1))

    response_1 = llm.invoke(messages)
    messages.append(AIMessage(content=response_1.content))
    print(f"ИИ: {response_1.content}")

    # Второе сообщение - проверяем память 
    user_input_2 = "Как меня зовут и что я изучаю?"
    messages.append(HumanMessage(content=user_input_2))

    response_2 = llm.invoke(messages)
    messages.append(response_2)
    print(f"ИИ: {response_2.content}")

    # Третье сообщение - продолжение темы
    user_input_3 = "Посоветуй мне книгу, курс или статью на эту тему"
    messages.append(HumanMessage(content=user_input_3))

    response_3 = llm.invoke(messages)
    print(f"ИИ: {response_3.content}")

    print(f"\nОбщее количество сообщений в истории: {len(messages)}")
    return messages


if __name__ == "__main__":
    chat_with_context()
