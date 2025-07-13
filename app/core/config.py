"""
Application configuration using Pydantic settings
"""

from typing import List, Optional

from pydantic import validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""

    # Application Settings
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model Configuration
    DEFAULT_MODEL: str = "medgemma-vision"
    MODEL_TIMEOUT: int = 30
    MAX_TOKENS: int = 1000
    TEMPERATURE: float = 0.7

    # Security
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: Optional[str] = None
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    # Docker Registry (for CI/CD)
    DOCKER_REGISTRY: str = "ghcr.io"
    DOCKER_IMAGE: str = "abdulrahmanalsumaih/openaimedvision"

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000

    @property
    def allowed_origins_list(self) -> List[str]:
        """Get allowed origins as a list"""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    @validator("DEBUG")
    def set_debug_based_on_environment(cls, v, values):
        """Set debug mode based on environment"""
        if "ENVIRONMENT" in values:
            return values["ENVIRONMENT"] == "development"
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
