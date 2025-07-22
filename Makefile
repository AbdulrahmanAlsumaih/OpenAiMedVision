# OpenAiMedVision Development Makefile

.PHONY: help dev build up down logs clean test format

# Default target
help:
	@echo "OpenAiMedVision Development Commands:"
	@echo ""
	@echo "  dev     - Start development environment with hot reload"
	@echo "  build   - Build development containers"
	@echo "  up      - Start services"
	@echo "  down    - Stop services"
	@echo "  logs    - View logs"
	@echo "  clean   - Clean up containers and volumes"
	@echo "  test    - Run tests"
	@echo "  format  - Format code"
	@echo "  shell   - Open shell in development container"

# Development environment
dev:
	@echo "🚀 Starting development environment..."
	docker-compose up --build

# Build containers
build:
	@echo "📦 Building containers..."
	docker-compose build

# Start services
up:
	@echo "🔧 Starting services..."
	docker-compose up -d

# Stop services
down:
	@echo "🛑 Stopping services..."
	docker-compose down

# View logs
logs:
	docker-compose logs -f openaimedvision-dev

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v --remove-orphans
	docker system prune -f

# Run tests
test:
	@echo "🧪 Running tests..."
	docker-compose exec openaimedvision-dev pytest

# Format code
format:
	@echo "✨ Formatting code..."
	docker-compose exec openaimedvision-dev black app/
	docker-compose exec openaimedvision-dev isort app/

# Open shell in container
shell:
	docker-compose exec openaimedvision-dev /bin/bash

# Health check
health:
	@echo "🏥 Checking service health..."
	curl -f http://localhost:8000/health || echo "Service not healthy" 