from typing import Any
from pydantic import BaseModel

class ToolResult(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    latency_ms: float = 0.0
    failure_mode: str | None = None  # e.g., "timeout", "empty", "malformed"
