from langgraph.graph import StateGraph, START, END

from app.multimodel.state import MultiModelState
from app.multimodel.nodes import (
    user_input_node,
    classify_task_node,
    code_analysis_node,
    dialog_response_node,
    local_context_node,
    synthesize_answer_node,
)
from app.multimodel.routing import (
    route_after_input,
    route_after_classification,
    route_continue,
)


def build_multi_model_graph():
    graph = StateGraph(MultiModelState)

    graph.add_node("get_input", user_input_node)
    graph.add_node("classify", classify_task_node)
    graph.add_node("analyze_code", code_analysis_node)
    graph.add_node("dialog_response", dialog_response_node)
    graph.add_node("local_context", local_context_node)
    graph.add_node("syntrsize", synthesize_answer_node)

    graph.add_edge(START, "get_input")

    graph.add_conditional_edges(
        "get_input",
        route_after_input,
        {
        "classify": "classify",
        "end": END,
        },
    )

    graph.add_edge("analyze_code", "synthesize")
    graph.add_edge("dialog_response", "synthesize")
    graph.add_edge("local_context", "synthesize")

    graph.add_conditional_edges(
        "synthesize",
        route_continue,
        {
            "get_input": "get_input",
            "end": END,
        },
    )

    return graph.compile()

    