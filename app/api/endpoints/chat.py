"""
Chat completions endpoint for OpenAI-compatible API
"""

import json
import time
from typing import List, Optional, Dict, Any, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, root_validator
from fastapi.responses import StreamingResponse
import asyncio

from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ImageUrl(BaseModel):
    """Image URL model for OpenAI format"""
    url: str = Field(
        ...,
        description="URL or base64 encoded image"
    )


class MessageContent(BaseModel):
    """Message content model for OpenAI format"""
    type: str = Field(
        ...,
        description="Content type: 'text' or 'image_url'"
    )
    text: Optional[str] = Field(
        None,
        description="Text content"
    )
    image_url: Optional[ImageUrl] = Field(
        None,
        description="Image URL"
    )


class Message(BaseModel):
    """Message model for OpenAI format - supports both text and vision formats"""
    role: str = Field(
        ...,
        description="Message role: 'user', 'assistant', or 'system'"
    )
    content: Union[str, List[MessageContent]] = Field(
        ...,
        description="Message content (string for text, list for vision)"
    )

    @root_validator(pre=True)
    def normalize_content(cls, values):
        content = values.get("content")
        if isinstance(content, str):
            values["content"] = [MessageContent(type="text", text=content)]
        return values


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""
    model: str = Field(
        default="medgemma-vision",
        description="Model to use"
    )
    messages: List[Message] = Field(
        ...,
        description="List of messages"
    )
    max_tokens: Optional[int] = Field(
        default=1000,
        description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=0.7,
        description="Sampling temperature"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response"
    )


class AssistantMessage(BaseModel):
    """Assistant message model for responses - uses text format"""
    role: str = Field(
        default="assistant",
        description="Message role"
    )
    content: str = Field(
        ...,
        description="Message content as text"
    )


class ChatCompletionChoice(BaseModel):
    """Chat completion choice model"""
    index: int
    message: AssistantMessage
    finish_reason: str


class Usage(BaseModel):
    """Token usage model"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


async def get_medgemma_response(
    messages: List[Message], max_tokens: int, temperature: float
) -> Dict[str, Any]:
    """Send request to MedGemma API and return response"""
    text_content = ""
    image_data = None
    for message in messages:
        if message.role == "user":
            for content in message.content:
                if content.type == "text" and content.text:
                    text_content += content.text + " "
                elif content.type == "image_url" and content.image_url:
                    if content.image_url.url.startswith("data:image"):
                        image_data = content.image_url.url.split(",")[1]
                    else:
                        image_data = content.image_url.url
    medgemma_payload = {
        "text": text_content.strip(),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if image_data:
        medgemma_payload["image"] = image_data
    logger.info(f"Sending request to MedGemma: {medgemma_payload}")
    text_part1 = "Analysis of the medical image: "
    text_part2 = text_content.strip()
    text_part3 = ". This is a mock response from MedGemma API."
    full_text = text_part1 + text_part2 + text_part3
    tokens_used = len(text_content.split()) + 50
    mock_response = {
        "text": full_text,
        "tokens_used": tokens_used,
    }
    return mock_response


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint compatible with OpenAI format, with streaming support"""
    try:
        logger.info(
            f"Received chat completion request for model: {request.model}"
        )
        if request.model != "medgemma-vision":
            logger.warning(
                f"Unsupported model requested: {request.model}"
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {request.model} not supported. Use 'medgemma-vision'"
                ),
            )
        medgemma_response = await get_medgemma_response(
            request.messages, request.max_tokens, request.temperature
        )
        if request.stream:
            async def event_stream():
                tokens = medgemma_response["text"].split()
                for i, token in enumerate(tokens):
                    chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": (
                                        " " + token
                                    ) if i > 0 else token
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.05)
                chunk_id = f"chatcmpl-{int(time.time())}"
                chunk_obj = "chat.completion.chunk"
                finish_reason = "stop"
                chunk = {
                    "id": chunk_id,
                    "object": chunk_obj,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(
                event_stream(), media_type="text/event-stream"
            )
        content_text = medgemma_response["text"]
        response_id = f"chatcmpl-{int(time.time())}"
        response = ChatCompletionResponse(
            id=response_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=AssistantMessage(
                        role="assistant",
                        content=content_text
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=medgemma_response.get("tokens_used", 100),
                completion_tokens=len(
                    medgemma_response["text"].split()
                ),
                total_tokens=(
                    medgemma_response.get("tokens_used", 100)
                    + len(
                        medgemma_response["text"].split()
                    )
                ),
            ),
        )
        logger.info("Successfully processed chat completion request")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
