from src.config.prompts import RAG_SYSTEM_PROMPT
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.core.llm import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.schemas.state import AgentState
from src.tools import web_search, python_sandbox, sql_lookup, self_reflection

class RAGTaskResult(BaseModel):
    task_id: str = Field(description="The ID of the task being solved.")
    answer: str = Field(description="The complete answer to the sub-task.")
    citations: List[str] = Field(description="A list of exact quotes or references used to form the answer.")
    tools_used: List[str] = Field(description="Names of tools used during multi-hop reasoning.")

@tool
async def web_search_tool(query: str) -> str:
    """Returns web search results. Use this to find live information from the internet."""
    result = await web_search(query)
    if not result.success:
        return json.dumps({"error": result.error, "failure_mode": result.failure_mode})
    return json.dumps(result.data)

@tool
async def python_sandbox_tool(code: str) -> str:
    """Executes Python snippets and returns stdout, stderr, and exit code. Use this for math or logic."""
    result = await python_sandbox(code)
    if not result.success:
        return json.dumps({"error": result.error, "failure_mode": result.failure_mode})
    return json.dumps(result.data)

@tool
async def sql_lookup_tool(sql_query: str) -> str:
    """Queries a local database via natural language converted to SQL."""
    result = await sql_lookup(sql_query)
    if not result.success:
        return json.dumps({"error": result.error, "failure_mode": result.failure_mode})
    return json.dumps(result.data)


async def rag_node(state: AgentState):
    """
    Multi-hop retrieval agent.
    Finds the next pending task, uses tools in a ReAct loop, and saves the result.
    """
    print("--- RAG NODE ---")
    sub_tasks = state.get("sub_tasks", {})
    completed_tasks = state.get("completed_task_results", {})
    
    # 1. Find the next pending task whose dependencies are met
    task_to_solve = None
    for task_id, task in sub_tasks.items():
        if task.status == "pending":
            # Check if all dependencies are in completed_tasks
            deps_met = all(dep in completed_tasks for dep in task.dependencies)
            if deps_met:
                task_to_solve = task
                break
                
    if not task_to_solve:
        print("No pending tasks with met dependencies found.")
        return {"next_node": "orchestrator_node"}
        
    print(f"Solving Task: {task_to_solve.task_id} - {task_to_solve.description}")
    
    llm = get_llm(temperature=0)
    
    system_msg = RAG_SYSTEM_PROMPT.format(
        task_id=task_to_solve.task_id,
        task_description=task_to_solve.description,
        completed_context=json.dumps(completed_tasks, indent=2)
    )
    
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content="Please begin solving the task. Use tools to gather information. When you have enough information, use the RAGTaskResult tool to submit your final answer.")
    ]
    
    # Bind our custom tools AND the Pydantic schema so the LLM can call it as a tool to finish
    llm_with_tools = llm.bind_tools([web_search_tool, python_sandbox_tool, sql_lookup_tool, RAGTaskResult])
    
    max_hops = 5
    final_result: RAGTaskResult | None = None
    
    for hop in range(max_hops):
        response = await llm_with_tools.ainvoke(messages)
        messages.append(response)
        
        if response.tool_calls:
            tool_call = response.tool_calls[0] # Take the first tool call
            tool_name = tool_call["name"]
            args = tool_call["args"]
            
            if tool_name == "RAGTaskResult":
                # The LLM decided it has the final answer!
                final_result = RAGTaskResult(**args)
                break
            else:
                # The LLM wants to use a real tool to gather data
                print(f"  -> RAG Agent executing tool: {tool_name}")
                
                try:
                    if tool_name == "web_search_tool":
                        res = await web_search_tool.ainvoke(args)
                    elif tool_name == "python_sandbox_tool":
                        res = await python_sandbox_tool.ainvoke(args)
                    elif tool_name == "sql_lookup_tool":
                        res = await sql_lookup_tool.ainvoke(args)
                    else:
                        res = f"Error: Unknown tool {tool_name}"
                except Exception as e:
                    res = f"Tool execution failed: {str(e)}"
                    
                messages.append(ToolMessage(content=str(res), tool_call_id=tool_call["id"]))
        else:
            # LLM didn't call any tools and output text directly. We force it to format.
            messages.append(HumanMessage(content="You must use the RAGTaskResult tool to submit your final answer formatted correctly."))
            
    if not final_result:
        # Fallback if we hit max hops without a structured response
        final_result = RAGTaskResult(
            task_id=task_to_solve.task_id,
            answer="Failed to solve task within tool hop limits or failed to parse output.",
            citations=[],
            tools_used=[]
        )
        
    print(f"Task {task_to_solve.task_id} solved.")
    if final_result.citations:
        print(f"Citations: {final_result.citations}")
    
    # Update the state
    task_to_solve.status = "completed"
    
    return {
        "sub_tasks": {**sub_tasks, task_to_solve.task_id: task_to_solve},
        "completed_task_results": {**completed_tasks, task_to_solve.task_id: final_result.answer},
        "next_node": "orchestrator_node"
    }
