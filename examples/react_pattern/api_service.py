"""Sketches for invoking an agent through HTTP/SSE handlers.

This file intentionally keeps framework-specific wiring out of the example.
Pass concrete ``agent``, ``config`` and ``format_sse`` objects from your app.
"""

from typing import Any, AsyncIterator, Callable

from langchain_core.messages import HumanMessage


async def invoke_agent(agent: Any, text: str, config: dict[str, Any]) -> dict[str, str]:
    """Return a fast single response from an async LangGraph agent."""
    result = await agent.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
    return {"reply": result["messages"][-1].content}


async def stream_agent_updates(
    agent: Any,
    text: str,
    config: dict[str, Any],
    format_sse: Callable[[Any], str],
) -> AsyncIterator[str]:
    """Yield serialized SSE chunks for real-time agent progress."""
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=text)]},
        config=config,
        stream_mode="updates",
    ):
        yield format_sse(chunk)
