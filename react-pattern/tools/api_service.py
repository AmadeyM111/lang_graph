# ainvoke (fast answer)

res = await agent.ainvoke({"messages": [HumanMessage(content=text)]}, config=config)
{"reply": res["messages"][-1].content}

# astream + SSE (progress in real-time)

async def stream():
    async for chunk in agent.astream({"messages": [HumanMessage(content=text)]},
                                    config=config,
                                    stream_mode="updates"):
        # сериализуете chunk и нправляем в клиент (SSE/WebSocket)
        yield format_sse(chunk)
    )
return EventSourceResponse(stream()) # sse-starlette / fastapi-sse