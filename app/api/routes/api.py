"""
Main API router configuration
"""

from fastapi import APIRouter

# Create main API router
api_router = APIRouter()


# For now, we'll add a simple test endpoint
@api_router.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"message": "OpenAiMedVision API is working!", "status": "success"}


# TODO: Add more endpoint routers here as we build them
# api_router.include_router(vision.router, prefix="/chat", tags=["chat"])
# api_router.include_router(models.router, prefix="/models", tags=["models"])
# api_router.include_router(health.router, prefix="/health", tags=["health"])
