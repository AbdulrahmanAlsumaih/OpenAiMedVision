"""
Vertex AI service for MedGemma-4b model integration
"""

import base64
import json
import subprocess
from typing import Dict, Any, Optional, List
import asyncio

import httpx

from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import PredictionServiceClient
from google.cloud.aiplatform_v1.types import PredictRequest, PredictResponse
from google.protobuf import json_format, struct_pb2

from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VertexAIService:
    """Service for interacting with Vertex AI MedGemma-4b model"""
    
    def __init__(self):
        self.project_id = settings.GOOGLE_CLOUD_PROJECT
        self.location = settings.GOOGLE_CLOUD_LOCATION
        self.endpoint_id = settings.VERTEX_AI_ENDPOINT_ID
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Vertex AI client"""
        try:
            # Initialize Vertex AI
            aiplatform.init(
                project=self.project_id,
                location=self.location
            )
            
            # Create prediction client
            self.client = PredictionServiceClient()
            logger.info(f"Vertex AI client initialized for project: {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI client: {str(e)}")
            # Don't raise here, let the service continue without the client
            # The HTTP requests will handle authentication directly
    
    def _get_endpoint_url(self) -> str:
        """Get the correct endpoint URL for the dedicated endpoint"""
        return f"https://{self.endpoint_id}.{self.location}-103350176976.prediction.vertexai.goog/v1/projects/{self.project_id}/locations/{self.location}/endpoints/{self.endpoint_id}:predict"
    
    def _prepare_medgemma_request(self, text: str, image_data: Optional[str] = None, 
                                 max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Prepare request payload for MedGemma-4b model"""
        
        # Prepare content array
        content = [{"type": "text", "text": text}]
        
        # Add image if provided
        if image_data:
            # Remove data URL prefix if present
            if image_data.startswith("data:image"):
                image_data = image_data.split(",", 1)[1]
            
            # Add image to content
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            })
        
        # Create the request in the correct format
        request_data = {
            "@requestFormat": "chatCompletions",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        return request_data
    
    async def predict_medgemma(self, text: str, image_data: Optional[str] = None,
                              max_tokens: int = 1000, temperature: float = 0.7) -> Dict[str, Any]:
        """Make prediction request to MedGemma-4b model"""
        
        try:
            # Prepare the request
            request_data = self._prepare_medgemma_request(
                text=text,
                image_data=image_data,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Create the full request structure
            full_request = {
                "instances": [request_data]
            }
            
            # Get the endpoint URL
            endpoint_url = self._get_endpoint_url()
            
            # Get the access token
            access_token = self._get_access_token()
            
            logger.info(f"Sending request to MedGemma-4b: {json.dumps(request_data, indent=2)}")
            
            # Make HTTP request
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    endpoint_url,
                    json=full_request,
                    headers={
                        "Authorization": f"Bearer {access_token}",
                        "Content-Type": "application/json"
                    },
                    timeout=60.0
                )
                
                response.raise_for_status()
                result = response.json()
            
            logger.info(f"MedGemma-4b response received: {json.dumps(result, indent=2)}")
            
            # Format the response to match our expected structure
            formatted_response = self._format_response(result, text)
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error calling MedGemma-4b: {str(e)}")
            raise
    
    def _get_access_token(self) -> str:
        """Get Google Cloud access token"""
        try:
            from google.auth import default
            from google.auth.transport.requests import Request
            
            # Get default credentials
            credentials, project = default()
            
            # Refresh the credentials to get a valid token
            credentials.refresh(Request())
            
            return credentials.token
        except Exception as e:
            logger.error(f"Failed to get access token via default credentials: {e}")
            # Try to use the mounted credentials file
            try:
                import json
                import os
                
                creds_path = "/app/gcloud-credentials.json"
                if os.path.exists(creds_path):
                    with open(creds_path, 'r') as f:
                        creds_data = json.load(f)
                    
                    # Check if this is a service account key or application default credentials
                    if 'client_email' in creds_data and 'token_uri' in creds_data:
                        # Service account key format
                        from google.oauth2 import service_account
                        credentials = service_account.Credentials.from_service_account_info(creds_data)
                    else:
                        # Application default credentials format
                        from google.oauth2 import credentials as oauth2_credentials
                        credentials = oauth2_credentials.Credentials.from_authorized_user_info(creds_data)
                    
                    credentials.refresh(Request())
                    return credentials.token
                else:
                    raise Exception(f"Credentials file not found at {creds_path}")
            except Exception as e2:
                logger.error(f"Failed to get access token via credentials file: {e2}")
                # Final fallback to gcloud command
                try:
                    result = subprocess.run(
                        ["gcloud", "auth", "print-access-token"],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    return result.stdout.strip()
                except subprocess.CalledProcessError as e3:
                    logger.error(f"Failed to get access token via gcloud: {e3}")
                    raise Exception(f"All authentication methods failed: {e}, {e2}, {e3}")
    
    def _format_response(self, vertex_response: Dict[str, Any], original_text: str) -> Dict[str, Any]:
        """Format Vertex AI response to match our expected structure"""
        
        # Extract the generated text from the response
        generated_text = ""
        actual_tokens = 0
        
        if isinstance(vertex_response, dict):
            # Handle the actual response format from MedGemma-4b
            if "predictions" in vertex_response:
                predictions = vertex_response["predictions"]
                if "choices" in predictions:
                    choices = predictions["choices"]
                    if choices and len(choices) > 0:
                        choice = choices[0]
                        if "message" in choice:
                            message = choice["message"]
                            if "content" in message:
                                generated_text = message["content"]
                
                # Extract actual token usage
                if "usage" in predictions:
                    usage = predictions["usage"]
                    actual_tokens = usage.get("total_tokens", 0)
            
            # Fallback to other possible formats
            elif "choices" in vertex_response:
                choices = vertex_response["choices"]
                if choices and len(choices) > 0:
                    choice = choices[0]
                    if "message" in choice:
                        message = choice["message"]
                        if "content" in message:
                            generated_text = message["content"]
            
            elif "text" in vertex_response:
                generated_text = vertex_response["text"]
            elif "generated_text" in vertex_response:
                generated_text = vertex_response["generated_text"]
            elif "content" in vertex_response:
                generated_text = vertex_response["content"]
            elif "response" in vertex_response:
                generated_text = vertex_response["response"]
            else:
                # Fallback: use the entire response as text
                generated_text = str(vertex_response)
        
        elif isinstance(vertex_response, str):
            generated_text = vertex_response
        
        else:
            generated_text = str(vertex_response)
        
        # Use actual tokens if available, otherwise estimate
        if actual_tokens > 0:
            tokens_used = actual_tokens
        else:
            tokens_used = len(original_text.split()) + len(generated_text.split())
        
        return {
            "text": generated_text,
            "tokens_used": tokens_used,
            "model": "medgemma-4b",
            "vertex_response": vertex_response  # Keep original for debugging
        }
    
    def health_check(self) -> bool:
        """Check if Vertex AI service is healthy"""
        try:
            return self.client is not None
        except Exception as e:
            logger.error(f"Vertex AI health check failed: {str(e)}")
            return False


# Global Vertex AI service instance
vertex_ai_service = VertexAIService() 