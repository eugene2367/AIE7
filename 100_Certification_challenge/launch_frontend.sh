#!/bin/bash

echo "🎬 Movie RAG Frontend Launcher"
echo "================================"

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed or not in PATH"
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Check if streamlit is available
if ! uv run python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing streamlit..."
    uv add streamlit
fi

# Check if the RAG system module exists
if [ ! -f "movie_rag_system.py" ]; then
    echo "❌ Error: movie_rag_system.py not found"
    echo "Please make sure you're in the correct directory"
    exit 1
fi

# Check for API keys
echo "🔑 Checking API keys..."
if ! uv run python setup_env.py check &> /dev/null; then
    echo ""
    echo "❌ API keys not configured!"
    echo "Please run the setup script first:"
    echo "   python setup_env.py"
    echo ""
    exit 1
fi

echo "✅ API keys configured!"
echo ""
echo "🚀 Starting Movie RAG Frontend..."
echo "💡 The system will initialize automatically!"
echo ""

# Launch the frontend
uv run streamlit run movie_rag_frontend.py 