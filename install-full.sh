#!/bin/bash
# LeanRAG Full Installation Script with nano-graphrag support
# This script creates a Python 3.10 environment for maximum compatibility
#
# Usage: ./install-full.sh

set -e

echo "🚀 LeanRAG Full Installation (Python 3.10 + nano-graphrag)"
echo "=========================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv package manager is required but not installed."
    echo "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

print_status "Creating Python 3.10 environment for maximum compatibility..."

# Create environment with Python 3.10
uv venv leanrag-full --python=3.10

if [[ $? -ne 0 ]]; then
    echo "❌ Failed to create Python 3.10 environment"
    echo "Make sure Python 3.10 is available: uv python install 3.10"
    exit 1
fi

# Activate environment
source leanrag-full/bin/activate

print_success "Python 3.10 environment created and activated ✓"

# Install everything
print_status "Installing LeanRAG with full dependencies..."
uv pip install -e .
uv pip install -e ".[nano-graphrag]"
uv pip install -e ".[full]"
uv pip install -e ".[google-cloud]"

print_success "Full installation completed! ✓"

echo ""
echo "🎉 LeanRAG Full Installation Complete!"
echo "======================================"
echo ""
echo "To use this environment:"
echo "  source leanrag-full/bin/activate"
echo ""
echo "This environment includes:"
echo "  ✅ LeanRAG core"
echo "  ✅ lightrag-hku"
echo "  ✅ nano-graphrag (Python 3.10 compatible)"
echo "  ✅ All ML/AI packages"
echo "  ✅ Google Cloud support"
echo ""
