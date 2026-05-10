from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from src.core.config import settings

def get_llm(temperature: float | None = None):
    """
    Centralized LLM factory driven by pydantic-settings.
    Supports easy switching between providers and models via .env.
    """
    temp = temperature if temperature is not None else settings.llm_temperature
    
    if settings.llm_provider.lower() == "groq":
        return ChatGroq(
            model=settings.llm_model, 
            temperature=temp,
            api_key=settings.groq_api_key
        )
    elif settings.llm_provider.lower() == "openai":
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temp,
            api_key=settings.openai_api_key
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

