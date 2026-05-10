import json
import uuid
import asyncio
import warnings
from typing import AsyncGenerator
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

# Suppress the upstream LangChain deprecation warning caused by LangGraph
from langchain_core._api.deprecation import LangChainPendingDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainPendingDeprecationWarning)

from src.agent.graph import graph
from src.schemas.state import AgentState
from src.core.context import ContextManager

app = FastAPI(
    title="Multi-Agent LLM Orchestrator",
    description="A real-time multi-agent system with SSE streaming.",
    version="0.1.0"
)

async def event_generator(query: str, job_id: str) -> AsyncGenerator[str, None]:
    """
    Generator that runs the LangGraph and yields SSE formatted strings.
    """
    # Initialize the starting state
    initial_state = {
        "job_id": job_id,
        "query": query,
        "messages": [],
        "sub_tasks": {},
        "completed_task_results": {},
        "critique_flags": [],
        # Setting a budget for demonstration
        "context_budget": {
            "rag_node": {"max_tokens": 4000, "used_tokens": 0},
            "orchestrator_node": {"max_tokens": 2000, "used_tokens": 0}
        },
        "tool_logs": [],
        "final_answer": None,
        "next_node": "orchestrator_node"
    }

    try:
        # We use astream_events (v2) to get granular node, tool, and LLM events
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event["event"]
            name = event.get("name", "")
            
            # 1. Node Transitions (Which agent is currently writing)
            if kind == "on_chain_start" and name.endswith("_node"):
                yield json.dumps({
                    "type": "agent_transition",
                    "agent": name,
                    "message": f"Agent '{name}' started execution."
                })
                yield "\n\n"

            # 2. Tool Calls In Flight
            elif kind == "on_tool_start":
                yield json.dumps({
                    "type": "tool_start",
                    "tool_name": name,
                    "input": event.get("data", {}).get("input")
                })
                yield "\n\n"
            
            elif kind == "on_tool_end":
                 yield json.dumps({
                    "type": "tool_end",
                    "tool_name": name,
                    "status": "completed"
                })
                 yield "\n\n"

            # 3. Token by Token Streaming
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    yield json.dumps({
                        "type": "token",
                        "content": chunk.content
                    })
                    yield "\n\n"
                    
            # 4. Graph End (Final State)
            elif kind == "on_chain_end" and name == "LangGraph":
                # The final event contains the final state of the graph
                final_state = event.get("data", {}).get("output", {})
                if final_state and isinstance(final_state, dict):
                    yield json.dumps({
                        "type": "final_result",
                        "final_answer": final_state.get("final_answer"),
                        "job_id": job_id
                    })
                    yield "\n\n"

    except Exception as e:
        yield json.dumps({
            "type": "error",
            "message": str(e)
        })
        yield "\n\n"


@app.post("/api/v1/query/stream")
async def submit_query_stream(request: Request):
    """
    Submit a query and receive a streaming SSE response with real-time agent activity.
    """
    body = await request.json()
    query = body.get("query")
    
    if not query:
        return {"error": "Query is required"}
        
    job_id = str(uuid.uuid4())
    
    # We use StreamingResponse with media_type="text/event-stream" to create the SSE connection.
    # We pass our async generator that runs the graph.
    return StreamingResponse(
        event_generator(query, job_id),
        media_type="text/event-stream"
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}
