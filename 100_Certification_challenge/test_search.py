#!/usr/bin/env python3
"""
Simple test script to debug the token limit issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from movie_rag_system import initialize_rag_system, query_movie_agent

def test_search():
    print("🧪 Testing search functionality...")
    
    # Initialize the system
    if not initialize_rag_system(max_movies=5000, min_reviews=5):
        print("❌ Failed to initialize system")
        return
    
    print("✅ System initialized successfully")
    
    # Test a simple query
    test_question = "What are the reviews on batman?"
    print(f"\n🔍 Testing query: {test_question}")
    
    try:
        result = query_movie_agent(test_question)
        print("✅ Query completed successfully")
        if isinstance(result, dict):
            print(f"Answer: {result.get('answer', 'No answer')[:500]}...")
        else:
            print(f"Answer: {str(result)[:500]}...")
    except Exception as e:
        print(f"❌ Error during query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search() 