#!/usr/bin/env python3
"""
OpenAiMedVision - Medical Vision API Gateway
Main application entry point
"""

import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.core.config import settings, print_environment_info
from app.api.routes.api import api_router
from app.utils.logger import setup_logging

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting OpenAiMedVision API Gateway...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Print environment configuration for debugging
    print_environment_info()
    
    yield
    
    # Shutdown
    logger.info("Shutting down OpenAiMedVision API Gateway...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="OpenAiMedVision",
        description="Medical Vision API Gateway with OpenAI-compatible endpoints",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router, prefix="/v1")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        from app.services.vertex_ai import vertex_ai_service
        
        # Check Vertex AI service health
        vertex_ai_healthy = vertex_ai_service.health_check()
        
        # Check authentication status
        auth_status = "unknown"
        auth_details = {}
        
        try:
            # Try to get an access token to test authentication
            access_token = vertex_ai_service._get_access_token()
            auth_status = "authenticated"
            auth_details = {"token_length": len(access_token) if access_token else 0}
        except Exception as e:
            auth_status = "failed"
            auth_details = {"error": str(e)}
            
            # Check for specific authentication issues
            if "gcloud" in str(e).lower():
                auth_details["issue"] = "gcloud_command_not_found"
                auth_details["solution"] = "Install Google Cloud SDK or use service account credentials"
            elif "credentials" in str(e).lower():
                auth_details["issue"] = "credentials_not_found"
                auth_details["solution"] = "Set GOOGLE_APPLICATION_CREDENTIALS or place credentials file"
        
        return {
            "status": "healthy" if vertex_ai_healthy else "degraded",
            "service": "OpenAiMedVision",
            "version": "1.0.0",
            "environment": settings.ENVIRONMENT,
            "components": {
                "vertex_ai": {
                    "status": "healthy" if vertex_ai_healthy else "unhealthy",
                    "authentication": {
                        "status": auth_status,
                        "details": auth_details
                    }
                }
            }
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "OpenAiMedVision API Gateway",
            "version": "1.0.0",
            "docs": "/docs" if settings.DEBUG else "Documentation disabled in production",
            "health": "/health",
            "auth_diagnostic": "/auth-diagnostic",
            "env_test": "/env-test"
        }
    
    # Environment test endpoint
    @app.get("/env-test")
    async def env_test():
        """Test endpoint to verify environment variables are loaded"""
        return {
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "google_cloud_project": settings.GOOGLE_CLOUD_PROJECT,
            "google_cloud_location": settings.GOOGLE_CLOUD_LOCATION,
            "vertex_ai_endpoint_id": settings.VERTEX_AI_ENDPOINT_ID,
            "google_application_credentials": settings.GOOGLE_APPLICATION_CREDENTIALS,
            "log_level": settings.LOG_LEVEL,
            "allowed_origins": settings.ALLOWED_ORIGINS
        }
    
    # Authentication diagnostic endpoint
    @app.get("/auth-diagnostic")
    async def auth_diagnostic():
        """Detailed authentication diagnostic endpoint"""
        from app.services.vertex_ai import vertex_ai_service
        import os
        
        diagnostic = {
            "timestamp": "2024-01-01T00:00:00Z",  # You can add datetime import if needed
            "environment": {
                "GOOGLE_CLOUD_PROJECT": settings.GOOGLE_CLOUD_PROJECT,
                "GOOGLE_CLOUD_LOCATION": settings.GOOGLE_CLOUD_LOCATION,
                "VERTEX_AI_ENDPOINT_ID": settings.VERTEX_AI_ENDPOINT_ID,
                "GOOGLE_APPLICATION_CREDENTIALS": settings.GOOGLE_APPLICATION_CREDENTIALS,
                "ENVIRONMENT": settings.ENVIRONMENT
            },
            "file_checks": {
                "credentials_file_exists": os.path.exists("/app/gcloud-credentials.json"),
                "credentials_file_readable": os.access("/app/gcloud-credentials.json", os.R_OK) if os.path.exists("/app/gcloud-credentials.json") else False,
                "env_credentials_exists": os.path.exists(settings.GOOGLE_APPLICATION_CREDENTIALS) if settings.GOOGLE_APPLICATION_CREDENTIALS else False
            },
            "command_checks": {
                "gcloud_available": False,
                "gcloud_version": None
            },
            "authentication_tests": {}
        }
        
        # Check if gcloud is available
        try:
            import subprocess
            result = subprocess.run(["gcloud", "--version"], capture_output=True, text=True)
            diagnostic["command_checks"]["gcloud_available"] = result.returncode == 0
            if result.returncode == 0:
                diagnostic["command_checks"]["gcloud_version"] = result.stdout.split('\n')[0]
        except (subprocess.CalledProcessError, FileNotFoundError):
            diagnostic["command_checks"]["gcloud_available"] = False
        
        # Test authentication methods
        try:
            access_token = vertex_ai_service._get_access_token()
            diagnostic["authentication_tests"]["success"] = True
            diagnostic["authentication_tests"]["token_length"] = len(access_token) if access_token else 0
        except Exception as e:
            diagnostic["authentication_tests"]["success"] = False
            diagnostic["authentication_tests"]["error"] = str(e)
            diagnostic["authentication_tests"]["error_type"] = type(e).__name__
        
        return diagnostic
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        # Don't handle HTTPException as it's already handled by FastAPI
        if isinstance(exc, HTTPException):
            raise exc
        
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc) if settings.DEBUG else "An error occurred"}
        )
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    ) 