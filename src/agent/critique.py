from src.config.prompts import CRITIQUE_PROMPT
import json
from typing import List
from pydantic import BaseModel, Field
from src.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

from src.schemas.state import AgentState, Critique

# Expected structured output
class CritiqueResult(BaseModel):
    critiques: List[Critique] = Field(
        description="A list of critiques, one for each completed task that hasn't been critiqued yet."
    )

async def critique_node(state: AgentState):
    """
    Reviews outputs, assigns confidence scores and flags spans.
    """
    print("--- CRITIQUE NODE ---")
    query = state.get("query", "")
    completed_tasks = state.get("completed_task_results", {})
    existing_critiques = state.get("critique_flags", [])
    
    # Find which tasks have already been critiqued
    critiqued_task_ids = {c.task_id for c in existing_critiques}
    
    # Find tasks that need critiquing
    tasks_to_critique = {}
    for task_id, result in completed_tasks.items():
        if task_id not in critiqued_task_ids:
            tasks_to_critique[task_id] = result
            
    if not tasks_to_critique:
        print("No new tasks to critique.")
        return {"next_node": "orchestrator_node"}
        
    print(f"Critiquing {len(tasks_to_critique)} tasks...")
    
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(CritiqueResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", CRITIQUE_PROMPT),
        ("human", "Please critique the provided tasks.")
    ])
    
    chain = prompt | structured_llm
    
    result: CritiqueResult = await chain.ainvoke({
        "query": query,
        "tasks_to_critique": json.dumps(tasks_to_critique, indent=2)
    })
    
    print(f"Generated {len(result.critiques)} critiques.")
    for critique in result.critiques:
        print(f" - Task {critique.task_id}: Score {critique.confidence_score}")
        if critique.flagged_spans:
            print(f"   Flags: {critique.flagged_spans}")
            
    # Return the new critiques. Because critique_flags is Annotated with operator.add,
    # returning a list here will append to the existing list in the state.
    return {
        "critique_flags": result.critiques,
        "next_node": "orchestrator_node"
    }
