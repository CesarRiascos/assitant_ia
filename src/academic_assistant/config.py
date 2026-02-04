"""Configuration settings for the Academic Assistant."""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class APIConfig(BaseModel):
    """Configuration for external APIs."""

    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    semantic_scholar_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    )
    crossref_email: Optional[str] = Field(
        default_factory=lambda: os.getenv("CROSSREF_EMAIL")
    )


class ModelConfig(BaseModel):
    """Configuration for AI models."""

    model_name: str = Field(default="gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1)


class PDFConfig(BaseModel):
    """Configuration for PDF processing."""

    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    max_pages: Optional[int] = Field(default=None)


class SearchConfig(BaseModel):
    """Configuration for academic search."""

    max_results: int = Field(default=10, ge=1, le=100)
    timeout: int = Field(default=30, ge=5)


class Settings(BaseModel):
    """Main settings class for the Academic Assistant."""

    api: APIConfig = Field(default_factory=APIConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        return cls()


# Global settings instance
settings = Settings.from_env()
