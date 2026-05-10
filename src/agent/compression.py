from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, RemoveMessage

from src.schemas.state import AgentState

COMPRESSION_PROMPT = """You are a Context Compression Agent.
Your job is to take a conversational history and compress it into a concise summary.

CRITICAL RULES:
1. You must be LOSSY for conversational filler (e.g., "Hello, I will now do this...", "Okay, next...").
2. You must be LOSSLESS for structured data (e.g., JSON, tool outputs, citations, exact scores). Do not modify or summarize structured data; copy it exactly.

Compress the following conversation history:
{text_to_compress}
"""

async def compression_node(state: AgentState):
    """
    Interceptor node that runs when context budget is near overflow.
    Compresses messages lossily, preserving structured data.
    """
    print("--- COMPRESSION NODE ---")
    messages = state.get("messages", [])
    
    if len(messages) <= 2:
        print("Not enough messages to compress.")
        return {"next_node": "orchestrator_node"}
        
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Prepare text to compress
    text_to_compress = "\n".join([f"{m.type}: {m.content}" for m in messages])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", COMPRESSION_PROMPT)
    ])
    
    chain = prompt | llm
    result = await chain.ainvoke({"text_to_compress": text_to_compress})
    
    print("Compression complete.")
    
    compressed_message = SystemMessage(content=f"--- COMPRESSED HISTORY ---\n{result.content}")
    
    # Remove old messages from the LangGraph state so we actually shrink the context window.
    # This requires messages to have IDs (which LangGraph automatically assigns).
    delete_messages = [RemoveMessage(id=m.id) for m in messages if m.id]
    
    return {
        # Return the delete commands followed by our new compressed summary
        "messages": delete_messages + [compressed_message],
        "next_node": "orchestrator_node"
    }
