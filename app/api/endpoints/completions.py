"""
Text completion endpoint for OpenAI/OpenRouter-compatible API
"""

import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

router = APIRouter()

class CompletionRequest(BaseModel):
    model: str = Field(default="medgemma-vision")
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class CompletionChoice(BaseModel):
    text: str
    index: int
    finish_reason: str

class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: CompletionUsage

@router.post("/", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    if request.model != "medgemma-vision":
        raise HTTPException(status_code=400, detail=f"Model {request.model} not supported. Use 'medgemma-vision'")
    # Mock completion
    completion_text = f"[MOCK COMPLETION] You said: {request.prompt}"
    now = int(time.time())
    return CompletionResponse(
        id=f"cmpl-{now}",
        created=now,
        model=request.model,
        choices=[
            CompletionChoice(
                text=completion_text,
                index=0,
                finish_reason="stop"
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=len(request.prompt.split()),
            completion_tokens=len(completion_text.split()),
            total_tokens=len(request.prompt.split()) + len(completion_text.split())
        )
    ) 