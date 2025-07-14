"""
Models endpoint for OpenAI-compatible API
"""

import time
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Any

router = APIRouter()


class ModelPermission(BaseModel):
    id: str = Field(default="modelperm-1")
    object: str = Field(default="model_permission")
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = True
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = Field(default="*")
    group: Any = None
    is_blocking: bool = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "organization-owner"
    permission: List[ModelPermission] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


@router.get("/", response_model=ModelListResponse)
async def list_models():
    model = ModelInfo(
        id="medgemma-vision",
        permission=[ModelPermission()]
    )
    return ModelListResponse(data=[model])
