from src.config.prompts import DECOMPOSITION_PROMPT
from typing import List
from pydantic import BaseModel, Field
from src.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate

from src.schemas.state import AgentState, SubTask

# Expected structured output
class DecompositionResult(BaseModel):
    tasks: List[SubTask] = Field(
        description="A list of distinct sub-tasks that, when completed in order, resolve the main query."
    )

async def decomposition_node(state: AgentState):
    """
    Breaks queries into a DAG of sub-tasks.
    """
    print("--- DECOMPOSITION NODE ---")
    query = state.get("query", "")
    
    if not query:
        print("No query found in state. Skipping decomposition.")
        return {"next_node": "orchestrator_node"}

    llm = get_llm(temperature=0)
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
