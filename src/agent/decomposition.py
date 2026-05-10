from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from src.schemas.state import AgentState, SubTask

# Expected structured output
class DecompositionResult(BaseModel):
    tasks: List[SubTask] = Field(
        description="A list of distinct sub-tasks that, when completed in order, resolve the main query."
    )

DECOMPOSITION_PROMPT = """You are the Decomposition Agent in a multi-agent system.
Your goal is to break down a complex or ambiguous user query into a Directed Acyclic Graph (DAG) of sub-tasks.

Rules:
1. Each task must have a unique `task_id` (e.g., "task_1", "task_2").
2. Each task must have a clear, actionable `description`.
3. If a task depends on the output of another task, list the parent's `task_id` in the `dependencies` array.
4. Independent tasks should have an empty `dependencies` array so they can be run in parallel.
5. Do not solve the query. Only plan how to solve it.
6. Keep the number of tasks to the minimum required.

Original Query:
{query}
"""

async def decomposition_node(state: AgentState):
    """
    Breaks queries into a DAG of sub-tasks.
    """
    print("--- DECOMPOSITION NODE ---")
    query = state.get("query", "")
    
    if not query:
        print("No query found in state. Skipping decomposition.")
        return {"next_node": "orchestrator_node"}

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_llm = llm.with_structured_output(DecompositionResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", DECOMPOSITION_PROMPT),
        ("human", "Decompose this query into sub-tasks.")
    ])
    
    chain = prompt | structured_llm
    
    result: DecompositionResult = await chain.ainvoke({"query": query})
    
    # Convert list of tasks into a dictionary keyed by task_id for easier state management
    task_dict = {task.task_id: task for task in result.tasks}
    
    print(f"Created {len(task_dict)} sub-tasks.")
    for task_id, task in task_dict.items():
        print(f" - {task_id}: {task.description} (Deps: {task.dependencies})")
    
    # Update the state with the new sub-tasks
    # LangGraph will replace the existing sub_tasks dict because we didn't use Annotated with operator.add for it.
    return {
        "sub_tasks": task_dict,
        "next_node": "orchestrator_node" # Always return control to orchestrator
    }
