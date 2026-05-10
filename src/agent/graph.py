from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.schemas.state import AgentState
from src.agent.orchestrator import orchestrator_node
from src.agent.decomposition import decomposition_node
from src.agent.rag import rag_node
from src.agent.critique import critique_node

# --- Node Skeleton Functions ---
# We will implement the actual LLM logic in these later.

async def synthesis_node(state: AgentState):
    """Merges outputs and resolves contradictions."""
    print("--- SYNTHESIS NODE ---")
    return {"final_answer": "Placeholder final answer", "next_node": END}

async def compression_node(state: AgentState):
    """Interceptor node that runs when context budget is near overflow."""
    print("--- COMPRESSION NODE ---")
    return state


# --- Edge Routing Logic ---

def orchestrator_router(state: AgentState) -> str:
    """
    Reads the 'next_node' decision from the orchestrator 
    and returns the string name of the node to execute next.
    """
    node = state.get("next_node")
    if not node:
        raise ValueError("Orchestrator failed to decide next node.")
    return node


# --- Graph Construction ---

def build_graph():
    builder = StateGraph(AgentState)

    # Add all agent nodes
    builder.add_node("orchestrator_node", orchestrator_node)
    builder.add_node("decomposition_node", decomposition_node)
    builder.add_node("rag_node", rag_node)
    builder.add_node("critique_node", critique_node)
    builder.add_node("synthesis_node", synthesis_node)
    builder.add_node("compression_node", compression_node)

    # Orchestrator is always the entry point
    builder.add_edge(START, "orchestrator_node")

    # The orchestrator dynamically routes to the next agent (or END)
    builder.add_conditional_edges(
        "orchestrator_node",
        orchestrator_router,
        {
            "decomposition_node": "decomposition_node",
            "rag_node": "rag_node",
            "critique_node": "critique_node",
            "synthesis_node": "synthesis_node",
            "compression_node": "compression_node",
            END: END
        }
    )

    # All sub-agents MUST return control to the orchestrator.
    # They do not talk to each other directly.
    builder.add_edge("decomposition_node", "orchestrator_node")
    builder.add_edge("rag_node", "orchestrator_node")
    builder.add_edge("critique_node", "orchestrator_node")
    builder.add_edge("synthesis_node", "orchestrator_node")
    builder.add_edge("compression_node", "orchestrator_node")

    return builder.compile()

# We can export the compiled graph
graph = build_graph()
