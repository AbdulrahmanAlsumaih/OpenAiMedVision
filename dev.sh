#!/bin/bash

# OpenAiMedVision Development Script

echo "🚀 Starting OpenAiMedVision Development Environment..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping development environment..."
    docker-compose down
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Build and start the development environment
echo "📦 Building development containers..."
docker-compose build

echo "🔧 Starting services..."
docker-compose up

# Keep script running
wait 