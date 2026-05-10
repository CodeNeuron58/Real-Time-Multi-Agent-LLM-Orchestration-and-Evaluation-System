from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.schemas.state import AgentState

# --- Node Skeleton Functions ---
# We will implement the actual LLM logic in these later.

async def orchestrator_node(state: AgentState):
    """
    The master orchestrator. Evaluates current state and decides the next step.
    Must output structured reasoning and set 'next_node' in the state.
    """
    print("--- ORCHESTRATOR NODE ---")
    # Placeholder logic: route to decomposition if no subtasks exist
    if not state.get("sub_tasks"):
        return {"next_node": "decomposition_node"}
    
    # If tasks exist but final answer doesn't, we are simplifying here for the skeleton
    if not state.get("final_answer"):
        return {"next_node": "synthesis_node"}
        
    return {"next_node": END}

async def decomposition_node(state: AgentState):
    """Breaks queries into a DAG of sub-tasks."""
    print("--- DECOMPOSITION NODE ---")
    return {"next_node": "orchestrator_node"}

async def rag_node(state: AgentState):
    """Multi-hop retrieval agent."""
    print("--- RAG NODE ---")
    return {"next_node": "orchestrator_node"}

async def critique_node(state: AgentState):
    """Reviews outputs, assigns confidence scores and flags spans."""
    print("--- CRITIQUE NODE ---")
    return {"next_node": "orchestrator_node"}

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
