"""
Test script for the Simple Agent Graph that interacts with the A2A protocol.

This script demonstrates how the Simple Agent can communicate with the existing
A2A protocol implementation through API calls.
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.simple_agent_graph import SimpleAgent


async def test_simple_agent():
    """Test the Simple Agent with various queries."""
    
    print("🤖 Simple Agent Graph - A2A Protocol Test")
    print("=" * 60)
    
    # Initialize the Simple Agent
    agent = SimpleAgent(a2a_url="http://localhost:10000")
    
    try:
        # Test different types of queries to demonstrate A2A interaction
        test_cases = [
            {
                "query": "What are the latest developments in artificial intelligence in 2024?",
                "description": "Web search query for current information"
            },
            {
                "query": "Find recent research papers on transformer architectures and their applications",
                "description": "Academic paper search query"
            },
            {
                "query": "What do the policy documents say about student loan requirements and eligibility?",
                "description": "Document retrieval query (if documents are loaded)"
            },
            {
                "query": "Compare the latest AI developments with academic research on neural networks",
                "description": "Multi-tool query requiring web search and academic papers"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🧪 Test Case {i}: {test_case['description']}")
            print("-" * 60)
            print(f"Query: {test_case['query']}")
            print("-" * 60)
            
            # Process the query
            result = await agent.process_query(test_case['query'])
            
            if result["success"]:
                print("✅ Success!")
                print("\n📝 FINAL RESPONSE:")
                print("=" * 50)
                print(result['response'])
                print("=" * 50)
                print(f"\n📊 A2A Interactions: {result['a2a_interactions']}")
                if result.get("thread_id"):
                    print(f"🔗 Thread ID: {result['thread_id']}")
            else:
                print("❌ Failed!")
                print(f"Error: {result['error']}")
            
            print("-" * 60)
            
            # Add a small delay between tests
            await asyncio.sleep(1)
        
        print("\n🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await agent.close()


async def test_conversation_flow():
    """Test a multi-turn conversation flow."""
    
    print("\n🔄 Testing Multi-Turn Conversation Flow")
    print("=" * 60)
    
    agent = SimpleAgent(a2a_url="http://localhost:10000")
    
    try:
        # Start a conversation
        initial_query = "What are the latest developments in AI?"
        print(f"Initial Query: {initial_query}")
        
        result1 = await agent.process_query(initial_query)
        if result1["success"]:
            print("\n📝 INITIAL RESPONSE:")
            print("=" * 50)
            print(result1['response'])
            print("=" * 50)
            
            # Follow-up question
            follow_up = "Can you provide more specific details about transformer models?"
            print(f"\n🔄 Follow-up: {follow_up}")
            
            result2 = await agent.process_query(follow_up)
            if result2["success"]:
                print("\n📝 FOLLOW-UP RESPONSE:")
                print("=" * 50)
                print(result2['response'])
                print("=" * 50)
            else:
                print(f"Follow-up failed: {result2['error']}")
        else:
            print(f"Initial query failed: {result1['error']}")
    
    except Exception as e:
        print(f"❌ Conversation test failed: {e}")
    
    finally:
        await agent.close()


def check_a2a_server():
    """Check if the A2A server is running."""
    import httpx
    
    try:
        # Try to access the agent card endpoint which should be available
        response = httpx.get("http://localhost:10000/v1/assistants", timeout=5)
        return response.status_code in [200, 404, 405]  # Any response means server is running
    except:
        return False


async def main():
    """Main test function."""
    
    print("🔍 Checking A2A server status...")
    if not check_a2a_server():
        print("❌ A2A server is not running on http://localhost:10000")
        print("Please start the A2A server first:")
        print("  uv run python -m app")
        print("  # or")
        print("  ./quickstart.sh")
        return
    
    print("✅ A2A server is running!")
    
    # Run the tests
    await test_simple_agent()
    await test_conversation_flow()


if __name__ == "__main__":
    asyncio.run(main()) 