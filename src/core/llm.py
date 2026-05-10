from langchain_groq import ChatGroq

def get_llm(temperature=0):
    """
    Centralized LLM factory.
    Using Llama 3 on Groq as the default execution model.
    """
    return ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=temperature
    )
