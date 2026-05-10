from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # LLM Settings
    llm_provider: Literal["groq", "openai"] = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    
    # API Keys (Pydantic automatically loads these from .env or OS env)
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    
    # Application Settings
    app_name: str = "Multi-Agent LLM Orchestrator"
    app_env: str = "development" # "development", "test", "production"

    # SettingsConfigDict tells Pydantic to read from the .env file
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore" # Ignore extra env vars not defined here
    )

# Instantiate a global settings object to be imported across the app
settings = Settings()
