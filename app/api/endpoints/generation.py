"""
Generation endpoint for OpenRouter-compatible API
"""

import time
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter()


class GenerationData(BaseModel):
    id: str
    total_cost: float = 1.1
    created_at: str = Field(
        default_factory=lambda: time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        )
    )
    model: str = "medgemma-vision"
    origin: str = "mock"
    usage: float = 1.1
    finish_reason: str = "stop"
    prompt: str = "This is a mock prompt."
    output: str = "This is a mock output."


class GenerationResponse(BaseModel):
    data: GenerationData


@router.get("/", response_model=GenerationResponse)
async def get_generation(id: str = Query(..., description="Generation ID")):
    return GenerationResponse(
        data=GenerationData(
            id=id,
            prompt="This is a mock prompt.",
            output="This is a mock output."
        )
    )
