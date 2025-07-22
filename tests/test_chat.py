"""
Tests for chat completions endpoint
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_chat_completions_endpoint():
    """Test chat completions endpoint with text only"""
    payload = {
        "model": "medgemma-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this medical image for any abnormalities."
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "medgemma-vision"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "content" in data["choices"][0]["message"]
    assert "usage" in data


def test_chat_completions_with_image():
    """Test chat completions endpoint with image"""
    # Create a simple base64 encoded image (1x1 pixel)
    base64_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    payload = {
        "model": "medgemma-vision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_image
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "medgemma-vision"


def test_chat_completions_invalid_model():
    """Test chat completions endpoint with invalid model"""
    payload = {
        "model": "invalid-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Test message"
                    }
                ]
            }
        ]
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400
    assert "not supported" in response.json()["detail"]


def test_chat_completions_missing_messages():
    """Test chat completions endpoint with missing messages"""
    payload = {
        "model": "medgemma-vision"
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422  # Validation error


def test_chat_completions_text_format():
    """Test chat completions endpoint with standard OpenAI text format"""
    payload = {
        "model": "medgemma-vision",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "max_tokens": 1000
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "medgemma-vision"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"


def test_chat_completions_mixed_formats():
    """Test chat completions endpoint with mixed text and vision formats"""
    payload = {
        "model": "medgemma-vision",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            },
            {
                "role": "assistant",
                "content": "I'm doing well, thank you!"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this medical image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "medgemma-vision" 


def test_list_models():
    """Test GET /v1/models returns the mock model list"""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
    assert any(model["id"] == "medgemma-vision" for model in data["data"]) 


 


def test_get_generation():
    """Test GET /v1/generation returns a mock generation"""
    gen_id = "mock-gen-id"
    response = client.get(f"/v1/generation?id={gen_id}")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert data["data"]["id"] == gen_id
    assert data["data"]["model"] == "medgemma-vision"
    assert data["data"]["output"] == "This is a mock output." 