#!/bin/bash
# CivicCam Deployment Script
# Run this on your VPS after cloning the repo

set -e

echo "🚀 CivicCam Deployment Script"
echo "=============================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "⚠️  Please log out and back in, then run this script again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo apt-get update
    sudo apt-get install -y docker-compose-plugin
fi

# Create required directories
mkdir -p data/evidence logs

# Build and run
echo "Building Docker image..."
docker compose build

echo "Starting CivicCam..."
docker compose up -d

echo ""
echo "✅ CivicCam is now running!"
echo ""
echo "📍 API: http://$(curl -s ifconfig.me):8000"
echo "📊 Health: http://$(curl -s ifconfig.me):8000/"
echo ""
echo "Useful commands:"
echo "  docker compose logs -f     # View logs"
echo "  docker compose restart     # Restart"
echo "  docker compose down        # Stop"
echo ""
