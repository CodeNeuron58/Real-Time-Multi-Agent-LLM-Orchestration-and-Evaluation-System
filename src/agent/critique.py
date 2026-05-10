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

CRITIQUE_PROMPT = """You are the Critique Agent in a multi-agent system.
Your job is to review the output of tasks completed by the RAG Agent.

For each completed task provided below, you must:
1. Assign a confidence score from 0.0 to 1.0.
2. Flag specific spans of text (exact quotes) that you disagree with, find factually suspicious, or that contradict other tasks. Do NOT flag the whole output, only the specific spans. If there are no issues, leave the flagged_spans list empty.
3. Provide constructive feedback explaining your score and flags.

Original Query:
{query}

Tasks to Critique:
{tasks_to_critique}

Remember, if a task introduces a contradiction with the premise of the query or another task, you MUST flag it so the Synthesis agent can resolve it later.
"""

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
