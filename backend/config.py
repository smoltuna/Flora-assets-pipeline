from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    database_url: str = "postgresql+asyncpg://flora:flora@localhost:5432/flora"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.2:3b"
    ollama_embed_model: str = "nomic-embed-text"

    # Cloud LLM providers
    groq_api_key: str = ""
    together_api_key: str = ""
    openai_api_key: str = ""

    llm_provider: str = "ollama"  # ollama | groq | together | openai

    # AWS
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket: str = "flora-assets"

    replicate_api_token: str = ""

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"

    # App
    log_level: str = "INFO"
    environment: str = "development"


settings = Settings()
