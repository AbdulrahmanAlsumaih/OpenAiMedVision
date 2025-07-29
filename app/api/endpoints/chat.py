"""
Chat completions endpoint for OpenAI-compatible API
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, root_validator

from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class ImageUrl(BaseModel):
    """Image URL model for OpenAI format"""

    url: str = Field(..., description="URL or base64 encoded image")


class MessageContent(BaseModel):
    """Message content model for OpenAI format"""

    type: str = Field(..., description="Content type: 'text' or 'image_url'")
    text: Optional[str] = Field(None, description="Text content")
    image_url: Optional[ImageUrl] = Field(None, description="Image URL")


class Message(BaseModel):
    """Message model for OpenAI format - supports both text and vision formats"""

    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: Union[str, List[MessageContent]] = Field(
        ..., description="Message content (string for text, list for vision)"
    )

    @root_validator(pre=True)
    def normalize_content(cls, values):
        content = values.get("content")
        if isinstance(content, str):
            values["content"] = [MessageContent(type="text", text=content)]
        return values


class ChatCompletionRequest(BaseModel):
    """Chat completion request model - supports all Open WebUI parameters"""

    model: str = Field(default="medgemma-4b-it", description="Model to use")
    messages: List[Message] = Field(..., description="List of messages")
    max_tokens: Optional[int] = Field(
        default=1000, description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=0.7, description="Sampling temperature"
    )
    top_p: Optional[float] = Field(
        default=1.0, description="Nucleus sampling parameter"
    )
    frequency_penalty: Optional[float] = Field(
        default=0.0, description="Frequency penalty"
    )
    presence_penalty: Optional[float] = Field(
        default=0.0, description="Presence penalty"
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default=None, description="Stop sequences"
    )
    n: Optional[int] = Field(
        default=1, description="Number of choices to generate"
    )
    stream: Optional[bool] = Field(
        default=False, description="Whether to stream the response"
    )


class AssistantMessage(BaseModel):
    """Assistant message model for responses - uses text format"""

    role: str = Field(default="assistant", description="Message role")
    content: str = Field(..., description="Message content as text")


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
    messages: List[Message], max_tokens: int, temperature: float,
    top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0,
    stop: Optional[Union[str, List[str]]] = None, n: int = 1
) -> Dict[str, Any]:
    """Send request to MedGemma-4b via Vertex AI and return response"""
    from app.services.vertex_ai import vertex_ai_service
    
    system_prompt = ""
    user_content = ""
    image_data = None
    
    # Process all messages to extract system prompt, user content, and images
    for message in messages:
        if message.role == "system":
            # Extract system prompt
            for content in message.content:
                if content.type == "text" and content.text:
                    system_prompt += content.text + " "
        elif message.role == "user":
            # Extract user content and images
            for content in message.content:
                if content.type == "text" and content.text:
                    user_content += content.text + " "
                elif content.type == "image_url" and content.image_url:
                    if content.image_url.url.startswith("data:image"):
                        image_data = content.image_url.url.split(",")[1]
                    else:
                        image_data = content.image_url.url
    
    # Clean up content
    system_prompt = system_prompt.strip()
    user_content = user_content.strip()
    
    # Combine system prompt with user content if system prompt exists
    if system_prompt:
        final_text = f"{system_prompt}\n\n{user_content}"
    else:
        final_text = user_content
    
    if not final_text:
        raise HTTPException(
            status_code=422,
            detail="No text content provided in the request"
        )
    
    logger.info(f"Preparing request for MedGemma-4b: system_prompt='{system_prompt[:50]}...', user_content='{user_content[:50]}...', has_image={image_data is not None}")
    
    try:
        # Call Vertex AI service with all parameters
        response = await vertex_ai_service.predict_medgemma(
            text=final_text,
            image_data=image_data,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            n=n
        )
        
        logger.info(f"MedGemma-4b response received: {len(response.get('text', ''))} characters")
        return response
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error calling MedGemma-4b via Vertex AI: {error_message}")
        
        # Check if this is an authentication error
        if "Authentication Error:" in error_message:
            # Extract the error details from the exception
            try:
                import ast
                # Find the error details in the message
                start_idx = error_message.find("{")
                if start_idx != -1:
                    error_json_str = error_message[start_idx:]
                    error_details = ast.literal_eval(error_json_str)
                    
                    # Create a user-friendly error message
                    user_error_message = f"""Google Cloud authentication failed. 

Error Details:
- Default credentials: {error_details.get('details', {}).get('default_credentials_error', 'Unknown')}
- Credentials file: {error_details.get('details', {}).get('credentials_file_error', 'Unknown')}
- gcloud command: {error_details.get('details', {}).get('gcloud_command_error', 'Unknown')}

Solutions:
{chr(10).join(f"- {solution}" for solution in error_details.get('solutions', []))}

Please set up Google Cloud authentication and try again."""
                else:
                    user_error_message = f"Authentication error: {error_message}"
            except:
                user_error_message = f"Authentication error: {error_message}"
        else:
            user_error_message = f"Vertex AI service error: {error_message}"
        
        # Return a fallback response with detailed error information
        fallback_text = f"Analysis of the medical image: {final_text}. This is a fallback response due to Vertex AI error: {user_error_message}"
        return {
            "text": fallback_text,
            "tokens_used": len(final_text.split()) + 50,
            "error": user_error_message,
            "error_type": "vertex_ai_error"
        }


@router.post("/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """Chat completions endpoint compatible with OpenAI format, with streaming support"""
    try:
        logger.info(f"Received chat completion request for model: {request.model}")
        if request.model != "medgemma-4b-it":
            logger.warning(f"Unsupported model requested: {request.model}")
            raise HTTPException(
                status_code=400,
                detail=(f"Model {request.model} not supported. Use 'medgemma-4b-it'"),
            )
        medgemma_response = await get_medgemma_response(
            request.messages, request.max_tokens, request.temperature,
            request.top_p, request.frequency_penalty, request.presence_penalty,
            request.stop, request.n
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
                                "delta": {"content": (" " + token) if i > 0 else token},
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

            return StreamingResponse(event_stream(), media_type="text/event-stream")
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
                    message=AssistantMessage(role="assistant", content=content_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=medgemma_response.get("tokens_used", 100),
                completion_tokens=len(medgemma_response["text"].split()),
                total_tokens=(
                    medgemma_response.get("tokens_used", 100)
                    + len(medgemma_response["text"].split())
                ),
            ),
        )
        logger.info("Successfully processed chat completion request")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat completions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
