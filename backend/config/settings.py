from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    API_KEY: str
    MODEL_NAME: str = "llama-3.1-70b-versatile"
    TEMPERATURE: float = 0
    CORS_ORIGINS: list[str] = ["*"]

    class Config:
        env_file = ".env"

@lru_cache
def get_settings():
    return Settings()
