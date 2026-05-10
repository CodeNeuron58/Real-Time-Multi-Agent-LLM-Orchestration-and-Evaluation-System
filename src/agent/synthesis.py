from src.config.prompts import SYNTHESIS_PROMPT
import json
from typing import List
from pydantic import BaseModel, Field
from src.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END

from src.schemas.state import AgentState

# Expected structured output
class Provenance(BaseModel):
    sentence: str = Field(description="A single sentence from the final answer.")
    source_task_id: str = Field(description="The ID of the task that provided the information.")
    source_chunk: str = Field(description="The specific chunk or citation from the task output that justifies this sentence.")

class SynthesisResult(BaseModel):
    final_answer: str = Field(
        description="The cohesive final answer addressing the user's original query."
    )
    provenance_map: List[Provenance] = Field(
        description="A map linking each sentence in the final answer to its source."
    )
    resolved_contradictions: List[str] = Field(
        description="A list explaining how any flagged contradictions were resolved."
    )

async def synthesis_node(state: AgentState):
    """
    Merges outputs, resolves contradictions, and generates provenance map.
    """
    print("--- SYNTHESIS NODE ---")
    query = state.get("query", "")
    completed_tasks = state.get("completed_task_results", {})
    critiques = state.get("critique_flags", [])
    
    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(SynthesisResult)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYNTHESIS_PROMPT),
        ("human", "Please synthesize the final answer.")
    ])
    
    chain = prompt | structured_llm
    
    # We serialize the lists/dicts to strings for the prompt
    critiques_json = json.dumps([c.model_dump() for c in critiques], indent=2)
    
    result: SynthesisResult = await chain.ainvoke({
        "query": query,
        "completed_tasks": json.dumps(completed_tasks, indent=2),
        "critiques": critiques_json
    })
    
    print("Synthesis complete.")
    if result.resolved_contradictions:
        print(f"Resolved Contradictions: {result.resolved_contradictions}")
        
    print("Final Answer Sample:", result.final_answer[:100], "...")
    
    # The final answer is now generated. We update the state.
    # We will just store the JSON string of the synthesis result in the state's final_answer field 
    # to capture the provenance map along with the text.
    
    return {
        "final_answer": result.model_dump_json(),
        "next_node": END  # We've reached the end of the pipeline
    }
