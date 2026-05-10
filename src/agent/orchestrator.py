from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from src.schemas.state import AgentState

# The expected structured output from the Orchestrator LLM
class OrchestratorDecision(BaseModel):
    next_node: Literal[
        "decomposition_node", 
        "rag_node", 
        "critique_node", 
        "synthesis_node", 
        "END"
    ] = Field(
        description="The exact name of the next node to route to. Use 'END' to finish execution."
    )
    justification: str = Field(
        description="Detailed reasoning for why this node was chosen based on the current state."
    )

# Orchestrator Prompt
ORCHESTRATOR_PROMPT = """You are the Master Orchestrator of a multi-agent system.
Your job is to examine the current state of the task and decide which specialized agent should act next.

Available Agents:
1. decomposition_node: Breaks down ambiguous or complex queries into a graph of sub-tasks. Route here first if sub_tasks is empty.
2. rag_node: Retrieves information to solve specific pending sub-tasks. Route here if there are pending tasks that need data.
3. critique_node: Reviews completed tasks to find errors, low confidence, or contradictions. Route here after RAG completes a task.
4. synthesis_node: Merges all critiqued and completed tasks into a final answer. Route here when all sub-tasks are complete and critiqued.
5. END: Route here ONLY if the final_answer has been generated and no further work is needed.

Current State Summary:
- Original Query: {query}
- Number of Sub-Tasks: {num_sub_tasks}
- Number of Completed Tasks: {num_completed_tasks}
- Number of Critiques: {num_critiques}
- Final Answer Exists: {has_final_answer}

Examine the state carefully. Output your decision and your justification.
"""

async def orchestrator_node(state: AgentState):
    """
    The master orchestrator. Evaluates current state and decides the next step via LLM.
    """
    print("--- ORCHESTRATOR NODE ---")
    
    # Initialize the LLM (ensure OPENAI_API_KEY is in env later)
    # Using a fast model for orchestration logic
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(OrchestratorDecision)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", ORCHESTRATOR_PROMPT),
        ("human", "What is the next step?")
    ])
    
    # Calculate state summaries for the prompt
    num_sub_tasks = len(state.get("sub_tasks", {}))
    num_completed_tasks = len(state.get("completed_task_results", {}))
    num_critiques = len(state.get("critique_flags", []))
    has_final_answer = state.get("final_answer") is not None

    chain = prompt | structured_llm
    
    decision: OrchestratorDecision = await chain.ainvoke({
        "query": state.get("query", ""),
        "num_sub_tasks": num_sub_tasks,
        "num_completed_tasks": num_completed_tasks,
        "num_critiques": num_critiques,
        "has_final_answer": has_final_answer
    })
    
    print(f"Decision: {decision.next_node}")
    print(f"Justification: {decision.justification}")
    
    # Return the state update. We map "END" from the LLM to LangGraph's END object string representation.
    next_route = END if decision.next_node == "END" else decision.next_node
    
    return {"next_node": next_route}
