import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Multi-Agent Document Assistant"
    DEBUG: bool = True
    
    # Paths
    UPLOAD_DIR: str = "/Users/saicherishnunna/Desktop/Multi Agent Document/backend/uploads"
    CHROMA_PERSIST_DIR: str = "/Users/saicherishnunna/Desktop/Multi Agent Document/backend/chroma_db"
    
    # API Keys
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    
    # DB Connections
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/document_assistant"
    REDIS_URL: str = "redis://localhost:6379"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure required directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
