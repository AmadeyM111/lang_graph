if __name__ == "__main__":
    print("Мультимодальная система техподдержки")
    print("DeepSeek - код | Amvera - диалоги | GigaChat - локальный контекст")
    print("Команда 'выход' для завершения")
    print("-" * 70)

    inintial_state = {
        "user_question": "",
        "task_type": "",
        "code_analysis": "",
        "dialog_response": "",
        "local_context": "",
        "final_answer": "",
        "should_continue": True
    }

    try:
        final_state = multi_model_app.invoke(initial_state)
        print("\n Система завершена!")

    except KeyboardInterrupt:
        print("\n\n Работа прервана (Ctrl+C)")
    except Exception as e:
        print(f"\n Ошибка системы: {e}")