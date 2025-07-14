"""
Main API router configuration
"""

from fastapi import APIRouter

from app.api.endpoints.chat import router as chat_router
from app.api.endpoints.models import router as models_router
from app.api.endpoints.completions import router as completions_router
from app.api.endpoints.generation import router as generation_router

# Create main API router
api_router = APIRouter()


# For now, we'll add a simple test endpoint
@api_router.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"message": "OpenAiMedVision API is working!", "status": "success"}


# Include chat router
api_router.include_router(chat_router, prefix="/chat", tags=["chat"])

# Include models router
api_router.include_router(models_router, prefix="/models", tags=["models"])

# Include completions router
api_router.include_router(
    completions_router, prefix="/completions", tags=["completions"]
)

# Include generation router
api_router.include_router(generation_router, prefix="/generation", tags=["generation"])

# TODO: Add more endpoint routers here as we build them
# api_router.include_router(models.router, prefix="/models", tags=["models"])
# api_router.include_router(health.router, prefix="/health", tags=["health"])
