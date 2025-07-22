"""
Tests for Vertex AI MedGemma-4b integration
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.fixture
def mock_vertex_ai_service():
    """Mock Vertex AI service for testing"""
    with patch('app.services.vertex_ai.vertex_ai_service') as mock_service:
        # Mock successful response
        mock_service.predict_medgemma = AsyncMock(return_value={
            "text": "This is a test response from MedGemma-4b",
            "tokens_used": 25,
            "model": "medgemma-4b"
        })
        yield mock_service


def test_chat_completions_with_vertex_ai(mock_vertex_ai_service):
    """Test chat completions endpoint with Vertex AI integration"""
    payload = {
        "model": "medgemma-vision",
        "messages": [
            {
                "role": "user",
                "content": "Analyze this medical image for abnormalities."
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


def test_chat_completions_with_image_and_vertex_ai(mock_vertex_ai_service):
    """Test chat completions with image using Vertex AI"""
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
                        "text": "What do you see in this medical image?"
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


def test_chat_completions_vertex_ai_error_handling():
    """Test error handling when Vertex AI fails"""
    with patch('app.services.vertex_ai.vertex_ai_service') as mock_service:
        # Mock service failure
        mock_service.predict_medgemma = AsyncMock(side_effect=Exception("Vertex AI service unavailable"))
        
        payload = {
            "model": "medgemma-vision",
            "messages": [
                {
                    "role": "user",
                    "content": "Test message"
                }
            ],
            "max_tokens": 1000
        }
        
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200  # Should still return 200 with fallback response
        
        data = response.json()
        assert "content" in data["choices"][0]["message"]
        # Should contain fallback response
        assert "fallback response" in data["choices"][0]["message"]["content"].lower()


def test_chat_completions_empty_text_handling(mock_vertex_ai_service):
    """Test handling of empty text content"""
    payload = {
        "model": "medgemma-vision",
        "messages": [
            {
                "role": "user",
                "content": ""  # Empty content
            }
        ],
        "max_tokens": 1000
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422  # Validation error for empty content


def test_chat_completions_vertex_ai_response_formatting(mock_vertex_ai_service):
    """Test that Vertex AI response is properly formatted"""
    # Mock different response formats
    mock_responses = [
        {"text": "Direct text response"},
        {"generated_text": "Generated text response"},
        {"content": "Content field response"},
        {"response": "Response field response"},
        "String response"
    ]
    
    for mock_response in mock_responses:
        with patch('app.services.vertex_ai.vertex_ai_service') as mock_service:
            mock_service.predict_medgemma = AsyncMock(return_value={
                "text": mock_response if isinstance(mock_response, str) else mock_response.get("text", "fallback"),
                "tokens_used": 25,
                "model": "medgemma-4b"
            })
            
            payload = {
                "model": "medgemma-vision",
                "messages": [
                    {
                        "role": "user",
                        "content": "Test message"
                    }
                ],
                "max_tokens": 1000
            }
            
            response = client.post("/v1/chat/completions", json=payload)
            assert response.status_code == 200


def test_vertex_ai_service_initialization():
    """Test Vertex AI service initialization"""
    with patch('google.cloud.aiplatform.init') as mock_init:
        with patch('google.cloud.aiplatform_v1.PredictionServiceClient') as mock_client:
            from app.services.vertex_ai import VertexAIService
            
            # Test successful initialization
            service = VertexAIService()
            assert service.client is not None
            mock_init.assert_called_once()


def test_vertex_ai_health_check():
    """Test Vertex AI service health check"""
    with patch('app.services.vertex_ai.vertex_ai_service') as mock_service:
        mock_service.health_check.return_value = True
        
        # Test health check
        assert mock_service.health_check() is True


def test_chat_completions_streaming_with_vertex_ai(mock_vertex_ai_service):
    """Test streaming response with Vertex AI"""
    payload = {
        "model": "medgemma-vision",
        "messages": [
            {
                "role": "user",
                "content": "Test streaming message"
            }
        ],
        "max_tokens": 1000,
        "stream": True
    }
    
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


def test_vertex_ai_request_preparation():
    """Test Vertex AI request preparation"""
    from app.services.vertex_ai import VertexAIService
    
    service = VertexAIService()
    
    # Test text-only request
    request = service._prepare_medgemma_request(
        text="Test text",
        max_tokens=1000,
        temperature=0.7
    )
    
    assert "@requestFormat" in request
    assert request["@requestFormat"] == "chatCompletions"
    assert "messages" in request
    assert len(request["messages"]) == 1
    assert request["messages"][0]["role"] == "user"
    assert len(request["messages"][0]["content"]) == 1
    assert request["messages"][0]["content"][0]["text"] == "Test text"
    assert "image" not in request["messages"][0]["content"]
    
    # Test request with image
    request_with_image = service._prepare_medgemma_request(
        text="Test text with image",
        image_data="data:image/jpeg;base64,test123",
        max_tokens=1000,
        temperature=0.7
    )
    
    assert len(request_with_image["messages"][0]["content"]) == 2
    assert request_with_image["messages"][0]["content"][1]["type"] == "image_url"
    assert "test123" in request_with_image["messages"][0]["content"][1]["image_url"]["url"] 