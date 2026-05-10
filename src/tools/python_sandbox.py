import asyncio
import sys
import io
import contextlib
import time
from src.tools.base import ToolResult

async def python_sandbox(code: str) -> ToolResult:
    """Runs Python snippets and returns stdout, stderr, and exit code."""
    start_time = time.time()
    
    # Very basic "sandbox" for demonstration. 
    # In a real scenario, use docker or restricted environments.
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    exit_code = 0
    
    try:
        async def _run_code():
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                try:
                    # pylint: disable=exec-used
                    exec(code, {"__builtins__": __builtins__}, {})
                except Exception as run_e:
                    print(str(run_e), file=sys.stderr)
                    return 1
            return 0
            
        # Hard timeout for code execution
        exit_code = await asyncio.wait_for(_run_code(), timeout=1.5)
        
        stdout_str = stdout_capture.getvalue()
        stderr_str = stderr_capture.getvalue()
        
        if exit_code != 0 or stderr_str:
            # Code ran but threw an error
            return ToolResult(
                success=False,
                error=stderr_str,
                data={"stdout": stdout_str, "exit_code": exit_code},
                failure_mode="malformed", # The agent wrote bad code
                latency_ms=(time.time() - start_time) * 1000
            )
            
        if not stdout_str.strip():
             return ToolResult(
                success=False,
                error="Code executed successfully but produced no stdout.",
                data={"exit_code": exit_code},
                failure_mode="empty",
                latency_ms=(time.time() - start_time) * 1000
            )

        return ToolResult(
            success=True,
            data={"stdout": stdout_str, "exit_code": exit_code},
            latency_ms=(time.time() - start_time) * 1000
        )
        
    except asyncio.TimeoutError:
        return ToolResult(
            success=False,
            error="Execution timed out. Potential infinite loop.",
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
