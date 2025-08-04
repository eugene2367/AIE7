#!/usr/bin/env python3
"""
Setup script for the Movie RAG Frontend
This script helps prepare the environment and run the frontend.
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'langchain_openai',
        'langchain_community',
        'langchain_cohere',
        'langchain_qdrant',
        'rank_bm25',
        'tavily',
        'ragas',
        'qdrant_client',
        'openai',
        'cohere',
        'tiktoken'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    return missing_packages

def install_dependencies():
    """Install missing dependencies"""
    print("\n🔧 Installing missing dependencies...")
    try:
        # Use uv to install dependencies
        subprocess.run([
            "uv", "add", "--dev", "streamlit"
        ], check=True)
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_rag_system():
    """Check if the RAG system is loaded"""
    print("\n🔍 Checking RAG system status...")
    
    try:
        from movie_rag_system import is_system_loaded, get_system_status
        
        if is_system_loaded():
            print("✅ RAG system is loaded and ready!")
            status = get_system_status()
            for component, loaded in status.items():
                status_icon = "✅" if loaded else "❌"
                print(f"   {status_icon} {component}")
            return True
        else:
            print("❌ RAG system not loaded")
            print("💡 You need to run the movie_reviews_rag_system.ipynb notebook first")
            return False
            
    except ImportError:
        print("❌ Could not import RAG system module")
        return False

def run_notebook():
    """Provide instructions for running the notebook"""
    print("\n📚 Notebook Setup Instructions:")
    print("=" * 50)
    print("1. Start Jupyter notebook:")
    print("   jupyter notebook")
    print()
    print("2. Open Movie_RAG_System.ipynb")
    print()
    print("3. Run ALL cells in the notebook (Cell → Run All)")
    print()
    print("4. Keep the notebook running (don't close it)")
    print()
    print("5. In a new terminal, run this setup script again")
    print("   python setup_frontend.py")
    print()
    print("6. Then start the frontend:")
    print("   streamlit run movie_rag_frontend.py")

def start_frontend():
    """Start the Streamlit frontend"""
    print("\n🚀 Starting Movie RAG Frontend...")
    try:
        subprocess.run([
            "uv", "run", "streamlit", "run", "movie_rag_frontend.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start frontend: {e}")
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")

def main():
    """Main setup function"""
    print("🎬 Movie RAG Frontend Setup")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    missing_packages = check_dependencies()
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        install_choice = input("Install missing dependencies? (y/n): ").lower().strip()
        if install_choice == 'y':
            if not install_dependencies():
                print("❌ Setup failed. Please install dependencies manually.")
                return
        else:
            print("❌ Setup cancelled. Please install dependencies manually.")
            return
    
    # Check RAG system
    rag_loaded = check_rag_system()
    
    if not rag_loaded:
        print("\n📚 RAG system needs to be loaded from the notebook")
        run_notebook()
        return
    
    # All good, start frontend
    print("\n🎉 Setup complete! Starting frontend...")
    start_frontend()

if __name__ == "__main__":
    main() 