from typing import Annotated, TypedDict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
import operator

class SubTask(BaseModel):
    task_id: str
    description: str
    dependencies: list[str] = Field(default_factory=list)
    status: str = Field(default="pending") # pending, in_progress, completed, failed

class Critique(BaseModel):
    task_id: str
    confidence_score: float
    flagged_spans: list[str] = Field(default_factory=list)
    feedback: str

class ToolCallLog(BaseModel):
    tool_name: str
    input: str
    output: str | None = None
    latency_ms: float = 0.0
    accepted: bool = False
    retry_count: int = 0

class BudgetInfo(BaseModel):
    max_tokens: int
    used_tokens: int = 0

class AgentState(TypedDict):
    """
    The shared state object that all agents interact with.
    We use Annotated with operator.add for lists so that returning a list from a node
    appends to the state rather than overwriting it.
    """
    job_id: str
    query: str
    messages: Annotated[list[BaseMessage], operator.add]
    sub_tasks: dict[str, SubTask]  # Dictionary for easier lookups by task_id
    completed_task_results: dict[str, str] # task_id -> result string
    critique_flags: Annotated[list[Critique], operator.add]
    context_budget: dict[str, BudgetInfo]
    tool_logs: Annotated[list[ToolCallLog], operator.add]
    final_answer: str | None
    next_node: str # Used by the orchestrator to route
