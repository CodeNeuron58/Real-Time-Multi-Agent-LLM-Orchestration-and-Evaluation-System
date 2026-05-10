import asyncio
import time
from src.tools.base import ToolResult

async def web_search(query: str) -> ToolResult:
    """A web search stub that returns structured results."""
    start_time = time.time()
    try:
        # Simulate network delay and potential timeout
        async def _search():
            await asyncio.sleep(0.5)
            if "fail" in query.lower():
                raise ValueError("Simulated network failure")
            if "empty" in query.lower():
                return []
            return [
                {"url": "https://example.com/1", "score": 0.95, "content": f"Result for {query}"},
                {"url": "https://example.com/2", "score": 0.82, "content": f"Another result for {query}"}
            ]

        results = await asyncio.wait_for(_search(), timeout=2.0)
        
        if not results:
            return ToolResult(
                success=False,
                error="Search returned no results.",
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
            error="Search timed out after 2 seconds.",
            failure_mode="timeout",
            latency_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=f"Search failed: {str(e)}",
            failure_mode="malformed",
            latency_ms=(time.time() - start_time) * 1000
        )
