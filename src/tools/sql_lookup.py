import asyncio
import time
from src.tools.base import ToolResult

async def sql_lookup(sql_query: str) -> ToolResult:
    """Queries a local database via natural language converted to SQL by the agent."""
    start_time = time.time()
    try:
        async def _run_query():
            await asyncio.sleep(0.1)
            # Stub logic
            if "DROP" in sql_query.upper():
                raise PermissionError("Destructive queries are not allowed.")
            if "EMPTY" in sql_query.upper():
                return []
            if "SYNTAX ERROR" in sql_query.upper():
                raise ValueError("Syntax error near 'ERROR'")
            return [{"id": 1, "value": "test data from DB"}]

        results = await asyncio.wait_for(_run_query(), timeout=1.0)
        
        if not results:
            return ToolResult(
                success=False,
                error="Query executed but returned 0 rows.",
                failure_mode="empty",
                latency_ms=(time.time() - start_time) * 1000
            )
            
        return ToolResult(
            success=True,
            data=results,
            latency_ms=(time.time() - start_time) * 1000
        )
        
    except asyncio.TimeoutError:
         return ToolResult(
            success=False,
            error="Database query timed out.",
            failure_mode="timeout",
            latency_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=str(e),
            failure_mode="malformed",
            latency_ms=(time.time() - start_time) * 1000
        )
