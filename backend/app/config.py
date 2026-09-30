from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration supporting environment variables and .env file."""
    
    # Core Application Settings
    APP_NAME: str = "Money Migration Atlas"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Offline-first demo mode & deterministic RNG
    DEMO_MODE: bool = True
    RANDOM_SEED: int = 42
    
    # Server Binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    
    # Graph Store Configuration
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "password"
    NEO4J_DATABASE: str = "neo4j"
    NEO4J_TIMEOUT_SECONDS: float = 3.0
    GRAPH_BACKEND: str = "networkx"
    NEO4J_FALLBACK_TO_NETWORKX: bool = True
    GRAPH_DEFAULT_HOPS: int = 3
    GRAPH_MAX_HOPS: int = 8

    # Blockchain Provider API Keys
    ETHERSCAN_API_KEY: Optional[str] = None
    BLOCKCHAIR_API_KEY: Optional[str] = None
    TRONGRID_API_KEY: Optional[str] = None

    # Fetching & Caching Configuration
    FETCH_TIMEOUT_SECONDS: int = 5
    CACHE_TTL_HOURS: int = 24
    PROVIDER_RETRY_ATTEMPTS: int = 2

    # Storage Paths
    DATA_DIR: str = "data"
    CACHE_DIR: str = "data/cache"

    # Machine Learning Configuration (Phase 4)
    ML_MODEL_PATH: str = "backend/data/models/xgboost_v1.pkl"
    ML_TRAIN_TEST_SPLIT: float = 0.15
    ML_RANDOM_SEED: int = 42
    ML_CONFIDENCE_HIGH_THRESHOLD: float = 0.75
    ML_CONFIDENCE_MEDIUM_THRESHOLD: float = 0.50
    ML_CONFIDENCE_LOW_THRESHOLD: float = 0.25

    # Agentic & LLM Configuration (Phase 7)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_TIMEOUT_SECONDS: int = 10
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    LLM_MAX_TOKENS: int = 400
    LLM_FALLBACK_ENABLED: bool = True

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
