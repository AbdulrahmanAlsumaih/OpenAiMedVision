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
    DEFAULT_MODEL: str = "medgemma-4b-it"
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
    
    # Google Cloud & Vertex AI
    GOOGLE_CLOUD_PROJECT: str = ""
    GOOGLE_CLOUD_LOCATION: str = "us-central1"
    VERTEX_AI_ENDPOINT_ID: str = ""
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None

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
    
    @validator("GOOGLE_CLOUD_PROJECT")
    def validate_google_cloud_project(cls, v):
        """Validate Google Cloud Project ID"""
        if not v:
            raise ValueError("GOOGLE_CLOUD_PROJECT is required")
        return v
    
    @validator("VERTEX_AI_ENDPOINT_ID")
    def validate_vertex_ai_endpoint_id(cls, v):
        """Validate Vertex AI Endpoint ID"""
        if not v:
            raise ValueError("VERTEX_AI_ENDPOINT_ID is required")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = "utf-8"


# Create settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings


def print_environment_info():
    """Print current environment configuration for debugging"""
    import os
    from app.utils.logger import get_logger
    
    logger = get_logger(__name__)
    
    logger.info("=== Environment Configuration ===")
    
    # Check required variables
    required_vars = [
        'ENVIRONMENT', 'DEBUG', 'GOOGLE_CLOUD_PROJECT', 
        'GOOGLE_CLOUD_LOCATION', 'VERTEX_AI_ENDPOINT_ID'
    ]
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var}: {value}")
        else:
            logger.warning(f"✗ {var}: Not set (required)")
    
    # Check optional variables
    optional_vars = [
        'GOOGLE_APPLICATION_CREDENTIALS', 'LOG_LEVEL', 
        'ALLOWED_ORIGINS', 'HOST', 'PORT'
    ]
    
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✓ {var}: {value}")
        else:
            logger.info(f"- {var}: Not set (optional)")
    
    logger.info("=== End Environment Configuration ===")
