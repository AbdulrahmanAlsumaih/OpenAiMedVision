# OpenAiMedVision

A medical-focused API router for vision-language models with OpenAI-compatible endpoints. This project serves as a backend service that routes medical image and text requests to various open-source or commercial vision LLM APIs (e.g., MedGemma, GPT-4 Vision, LLaVA, etc.).

---

## 🏥 Project Overview

OpenAiMedVision is a unified API gateway that provides medical professionals and researchers with easy access to state-of-the-art vision-language models for medical image analysis. The service follows OpenAI's API format, making it a drop-in replacement for existing medical AI workflows.

**This project does NOT run models locally. It acts as a router/gateway to external LLM APIs.**

### Key Features
- **🔄 Unified API Endpoint**: Single endpoint compatible with OpenAI's vision API format
- **🏥 Medical-Focused**: Optimized for medical image analysis and text processing
- **🔌 Model Router**: Routes requests to different vision LLM APIs (MedGemma, GPT-4 Vision, LLaVA, etc.)
- **🐳 Docker-First Development**: All development and deployment is done in Docker
- **📊 Flexible Input**: Handles both image and text inputs seamlessly (no local image processing)
- **🔒 Production Ready**: Built with FastAPI for high performance and scalability

## 🚀 Supported Models/APIs
- **MedGemma Vision-Text API**
- **GPT-4 Vision API**
- **LLaVA API**
- (Pluggable for more in the future)

## 🛠️ Tech Stack
- **Backend Framework**: FastAPI (Python)
- **API Routing**: httpx, requests
- **Containerization**: Docker, Docker Compose
- **Dev Tools**: Makefile, dev.sh, pytest, black, flake8, mypy
- **Monitoring**: structlog, prometheus-client

## 📁 Project Structure
```
OpenAiMedVision/
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── api.py             # API router configuration
│   ├── core/
│   │   └── config.py              # Application configuration
│   └── utils/
│       └── logger.py              # Logging utilities
├── Dockerfile                     # Production Dockerfile
├── Dockerfile.dev                 # Development Dockerfile (hot reload)
├── docker-compose.yml             # Multi-service setup
├── dev.sh                         # Development helper script
├── Makefile                       # Common dev commands
├── requirements.txt
├── main.py                        # FastAPI entrypoint
└── README.md
```

## 🚀 Quick Start (Docker-First)

### Prerequisites
- Docker & Docker Compose
- Git

### Development Workflow

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbdulrahmanAlsumaih/OpenAiMedVision.git
   cd OpenAiMedVision
   ```

2. **Start the development environment**
   ```bash
   ./dev.sh
   # or
   make dev
   ```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

4. **Common Commands**
   ```bash
   make up        # Start services in background
   make down      # Stop services
   make logs      # View logs
   make test      # Run tests
   make lint      # Run linting
   make format    # Format code
   make shell     # Open shell in dev container
   ```

### Production Build

```bash
# Build and run with Docker
make build
make up
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory (or use `.env.dev` for development):

```env
# Application Settings
ENVIRONMENT=development
DEBUG=true
HOST=0.0.0.0
PORT=8000

# Model Routing (example)
DEFAULT_MODEL=medgemma-vision
MODEL_TIMEOUT=30

# Security
API_KEY_HEADER=X-API-Key
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000

# Docker Registry (for CI/CD)
DOCKER_REGISTRY=ghcr.io
DOCKER_IMAGE=abdulrahmanalsumaih/openaimedvision
```

## 📚 API Usage

### OpenAI-Compatible Endpoint
The main endpoint follows OpenAI's vision API format:

```
POST /v1/chat/completions
```

### Example Request
```python
import requests
import base64

# Encode image to base64
with open("medical_image.jpg", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

payload = {
    "model": "medgemma-vision",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this chest X-ray for any abnormalities."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }
    ],
    "max_tokens": 1000
}

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json=payload,
    headers={"Content-Type": "application/json"}
)
print(response.json())
```

### Example Response
```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "medgemma-vision",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Analysis of the chest X-ray reveals..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300
    }
}
```

## 🧪 Testing

```bash
make test
# or
make lint
# or
make format
```

## 🔒 Security & Compliance
- **API Key Authentication**: Secure access control
- **Input Validation**: Comprehensive request validation
- **Rate Limiting**: Protection against abuse
- **CORS Configuration**: Controlled cross-origin access
- **Medical Data Handling**: HIPAA-compliant practices

## 🤝 Contributing
- Fork the repository
- Create a feature branch (`git checkout -b feature/amazing-feature`)
- Commit your changes (`git commit -m 'Add amazing feature'`)
- Push to the branch (`git push origin feature/amazing-feature`)
- Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation for API changes
- Use conventional commit messages

## 📄 License
MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support
- **Issues**: Create an issue on GitHub
- **Docs**: Check the `/docs` endpoint when running locally
- **Contact**: abdulrahman.alsumaih@icloud.com

---

**⚠️ Disclaimer**: This is a research and development project. Not intended for clinical use without proper validation and regulatory approval. Always consult with healthcare professionals for medical decisions. 