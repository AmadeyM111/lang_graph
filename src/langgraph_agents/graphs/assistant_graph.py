def code_analyze_node(state: MultiModelState) -> dict:
    """Узел анализа кода - специализация DeepSeek"""
    question = state["user_question"]

    try:
        print("DeepSeek: Анализирую код...")

        code messages = [
            SystemMessage(content="""Ты эксперт-программист из META. Анализируй код, находи ошибки, предлагай оптимизации. Отвечай технично и точно.""")
            HumanMessage(content=question)
        ]

        response = deepseek_model.invoke(code_messages)
        analysis = response.content

        print(f"DeepSeek: Анализ кода: {analysis[:100]}...")

        return {"code_analysis": analysis}

    except Exception as e:
        print(f"Ошибка анализа кода: {e}")
        return {"code_analysis": "Извините, произошла ошибка при анализе кода."}

# --------------- Узел генерации ответа для диалога ---------------

def dialog_response_node(state: MultiModelStage)-> dict:
    """Узел генерации ответа для диалога - используем LLaMA"""
    question = state["user_question"]

    try:
        print("LLaMA: Генерирую ответ для диалога...")

        dialog_messages = [
            SystemMessage(content="""Ты дружелюбный помощник. Отвечай развернуто, помогай, объясняй.""")
            HumanMessage(content=question)
        ]
        
        response = amvera_model.invoke(dialog_messages)
        answer = response.content

        print(f"LLaMA: Ответ для диалога: {dialog_answer[:100]}...")

        return {"dialog_answer": answer}

    except Exception as e:
        print(f"Ошибка генерации ответа для диалога: {e}")
        return {"dialog_answer": "Извините, произошла ошибка при генерации ответа для диалога."}

# --------------- Узел локального контекста ---------------

def local_context_node(state: MultiModelStage)-> dict:
    """Узел локального контекста - эуспертиза GigaChat"""
    question = state["user_question"]

    try:
        print("GigaChat: анализирует локальный контекст...")

        dialog_messages = [
            SystemMessage(content="""Ты эксперт по России: законы, традиции, особенности, госуслуги, местная специфика. Давай точную информацию о российских реалиях."""),
            HumanMessage(content=question)
        ]

        response = gigachat_model.invoke(local_messages)
        local_info = response.content

        print(f"GigaChat: {local_info[:100]}...")

        return {"local_context": local_info}

    except Exception as e:
        print(f"Ошибка gigachat: {e}")
        return {"dialog_context": "Ошибка анализа локального контекста."}

# ------------- Узел получения пользовательского ввода ------------------------

def user_input_node(state: MultiModelState) -> dict:
    """ Узел получения вопроса от пользователя """
    question = input("\n Ваш вопрос:").strip()

    if question.lower() in ["выход", "quit", "exit", "bye"]:
        return {"should_continue": False}

    return {
        "user_question": question,
        "should_continue": True
    }

# --------------- Узел синтеза финального ответа ---------------------

def synthesize_answer_node(state: MultiModelState) -> dict:
    """Узел синтеза финального ответа - используем Ollama для объединения"""
    task_type = state["task_type"]
    question = state["user_question"]

    # Собираем доступные результаты
    results = []

    if state.get("code_analysis"):
        results.append(f"Технический анализ: {state['code_analysis']}")

    if state.get("dialog_response"):
        results.append(f"Общий ответ: {state['dialog_response']}")

    if state.get("local_context"):
        results.append(f"Локальная информация: {state['local_context']}")

    if not results:
        return {"final_answer": "Не удалось получить ответ от моделей"}

    try:
        print("Синтезирую итоговый ответ...")

        synthesis_prompt = f"""На основе результатов от разных ИИ-моделей дай пользователю единый полезный ответ.

Вопрос пользователя: {question}
Тип задачи: {task_type}

Результаты от моделей:
{chr(10).join(results)}

Создай связный, полезный ответ, объединив лучшее из каждого источника."""

        synthesis_messages = [
            SystemMessage(content="Ты синтезируешь ответы от разных ИИ в единый полезный ответ."),
            HumanMessage(content=synthesis_prompt)
        ]

        response = amvera_model.invoke(synthesis_messages)
        final_answer = response.content

        print("="*60)
        print("ИТОГОВЫЙ ОТВЕТ:")
        print("="*60)
        print(final_answer)
        print("="*60)

        return {"final_answer": final_answer}

    except Exception as e:
        print(f"Ошибка синтеза: {e}")
        return {"final_answer": "Ошибка при создании итогового ответа"}