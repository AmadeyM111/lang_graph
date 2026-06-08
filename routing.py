def route_after_input(state: MultiModelState) -> str:
    """ routing after input """
    if not state.get("should_continue", True):
        return "end"
    return "classify"

def route_after_classification(state: MultiModelState) -> str:
    """ task-type routing """
    task_type = state.get("task_type", "dialog")

    if task_type == "code":
        return "analyze_code"
    elif task_type == "local":
        return "local_context"
    else:
        return "dialog_response"

def route_to_synthesis(state: MultiModelState) -> str:
    """ Маршрутизация к синтезу ответа """
    return "synthesize"

def route_continue(state: MultiModelState) -> str:
    """ Проверка продолжения """
    return "get_input" if state.get("should_continue", True) else "end"
