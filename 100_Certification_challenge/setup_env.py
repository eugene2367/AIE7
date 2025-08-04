#!/usr/bin/env python3
"""
Setup script for Movie RAG System API Keys
This script helps you configure your API keys for the movie RAG system.
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create a .env file with user input"""
    
    print("🎬 Movie RAG System - API Key Setup")
    print("=" * 50)
    print()
    
    # Check if .env already exists
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("Setup cancelled.")
            return
    
    # Get API keys from user
    print("Please enter your API keys (press Enter to skip optional ones):")
    print()
    
    # Required: OpenAI API Key
    openai_key = input("🔑 OpenAI API Key (REQUIRED): ").strip()
    if not openai_key:
        print("❌ OpenAI API Key is required!")
        print("Get your key from: https://platform.openai.com/api-keys")
        return
    
    # Optional: Cohere API Key
    cohere_key = input("🔑 Cohere API Key (optional, for reranking): ").strip()
    
    # Optional: Tavily API Key
    tavily_key = input("🔑 Tavily API Key (optional, for external search): ").strip()
    
    # Optional: LangSmith API Key
    langsmith_key = input("🔑 LangSmith API Key (optional, for tracing): ").strip()
    
    # Create .env content
    env_content = f"""# OpenAI API Key (Required)
OPENAI_API_KEY={openai_key}

# Cohere API Key (Optional - for reranking)
COHERE_API_KEY={cohere_key or 'your_cohere_api_key_here'}

# Tavily API Key (Optional - for external search)
TAVILY_API_KEY={tavily_key or 'your_tavily_api_key_here'}

# LangSmith API Key (Optional - for tracing and monitoring)
LANGSMITH_API_KEY={langsmith_key or 'your_langsmith_api_key_here'}
LANGSMITH_PROJECT=Movie-Reviews-RAG

# Optional: Set to true to enable detailed logging
DEBUG=false
"""
    
    # Write .env file
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print()
        print("✅ .env file created successfully!")
        print()
        print("📋 API Keys configured:")
        print(f"   ✅ OpenAI: {'*' * (len(openai_key) - 4) + openai_key[-4:] if len(openai_key) > 4 else '***'}")
        if cohere_key:
            print(f"   ✅ Cohere: {'*' * (len(cohere_key) - 4) + cohere_key[-4:] if len(cohere_key) > 4 else '***'}")
        if tavily_key:
            print(f"   ✅ Tavily: {'*' * (len(tavily_key) - 4) + tavily_key[-4:] if len(tavily_key) > 4 else '***'}")
        if langsmith_key:
            print(f"   ✅ LangSmith: {'*' * (len(langsmith_key) - 4) + langsmith_key[-4:] if len(langsmith_key) > 4 else '***'}")
        
        print()
        print("🚀 You can now run the frontend:")
        print("   uv run streamlit run movie_rag_frontend.py")
        
    except Exception as e:
        print(f"❌ Error creating .env file: {e}")
        return

def check_env_file():
    """Check if .env file exists and has required keys"""
    env_file = Path(".env")
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print("Run this script to set up your API keys:")
        print("   python setup_env.py")
        return False
    
    # Load and check .env file
    try:
        with open(".env", "r") as f:
            content = f.read()
        
        if "OPENAI_API_KEY=your_openai_api_key_here" in content:
            print("⚠️  OpenAI API key not configured!")
            print("Please update your .env file with your actual API key.")
            return False
        
        if "OPENAI_API_KEY=" in content:
            print("✅ .env file found with API keys configured!")
            return True
        else:
            print("❌ OpenAI API key missing from .env file!")
            return False
            
    except Exception as e:
        print(f"❌ Error reading .env file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_env_file()
    else:
        create_env_file() 