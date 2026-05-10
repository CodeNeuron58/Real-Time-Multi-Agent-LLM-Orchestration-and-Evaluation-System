import time
from typing import Any, Dict
from src.tools.base import ToolResult

async def self_reflection(task_id: str, current_state_dict: Dict[str, Any]) -> ToolResult:
    """Agent calls this to re-read its own previous outputs within the session."""
    start_time = time.time()
    try:
        completed = current_state_dict.get("completed_task_results", {})
        if task_id not in completed:
            return ToolResult(
                success=False,
                error=f"Task {task_id} not found in completed tasks.",
                failure_mode="empty",
                latency_ms=(time.time() - start_time) * 1000
            )
            
        return ToolResult(
            success=True,
            data={"task_content": completed[task_id]},
            latency_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=str(e),
            failure_mode="malformed",
            latency_ms=(time.time() - start_time) * 1000
        )
