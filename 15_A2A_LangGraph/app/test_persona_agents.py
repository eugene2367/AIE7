"""
Test script for the Persona Agents that demonstrate different agent personas using the A2A protocol.

This script showcases how different agent personas can interact with the existing
A2A protocol implementation, each with their own goals and communication styles.
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

from app.persona_agent_graph import PersonaAgent


async def test_ml_expert_persona():
    """Test the Machine Learning Expert persona."""
    print("🧠 Testing Machine Learning Expert Persona")
    print("=" * 60)
    
    # Initialize the ML Expert persona
    agent = PersonaAgent("Machine Learning Expert")
    
    try:
        # Test query
        test_query = "What makes Kimi K2 so incredible? I want technical details and academic sources."
        print(f"Query: {test_query}")
        print(f"Persona Goals: {', '.join(agent._get_persona_config()['goals'])}")
        print(f"Persona Style: {agent._get_persona_config()['style']}")
        print("-" * 60)
        
        result = await agent.process_query(test_query)
        
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
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # No cleanup needed for the model
        pass
    
    print("-" * 60)

async def test_business_analyst_persona():
    """Test the Business Analyst persona."""
    print("💼 Testing Business Analyst Persona")
    print("=" * 60)
    
    # Initialize the Business Analyst persona
    agent = PersonaAgent("Business Analyst")
    
    try:
        # Test query
        test_query = "What are the latest developments in artificial intelligence?"
        print(f"Query: {test_query}")
        print(f"Persona Goals: {', '.join(agent._get_persona_config()['goals'])}")
        print(f"Persona Style: {agent._get_persona_config()['style']}")
        print("-" * 60)
        
        result = await agent.process_query(test_query)
        
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
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # No cleanup needed for the model
        pass
    
    print("-" * 60)

async def test_curious_student_persona():
    """Test the Curious Student persona."""
    print("🎓 Testing Curious Student Persona")
    print("=" * 60)
    
    # Initialize the Curious Student persona
    agent = PersonaAgent("Curious Student")
    
    try:
        # Test query
        test_query = "What are the latest developments in artificial intelligence?"
        print(f"Query: {test_query}")
        print(f"Persona Goals: {', '.join(agent._get_persona_config()['goals'])}")
        print(f"Persona Style: {agent._get_persona_config()['style']}")
        print("-" * 60)
        
        result = await agent.process_query(test_query)
        
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
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # No cleanup needed for the model
        pass
    
    print("-" * 60)

async def test_skeptical_reviewer_persona():
    """Test the Skeptical Reviewer persona."""
    print("🔍 Testing Skeptical Reviewer Persona")
    print("=" * 60)
    
    # Initialize the Skeptical Reviewer persona
    agent = PersonaAgent("Skeptical Reviewer")
    
    try:
        # Test query
        test_query = "What are the latest developments in artificial intelligence?"
        print(f"Query: {test_query}")
        print(f"Persona Goals: {', '.join(agent._get_persona_config()['goals'])}")
        print(f"Persona Style: {agent._get_persona_config()['style']}")
        print("-" * 60)
        
        result = await agent.process_query(test_query)
        
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
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # No cleanup needed for the model
        pass
    
    print("-" * 60)


async def test_persona_comparison():
    """Test how different personas handle the same query."""
    print("\n🔄 Testing Persona Comparison - Same Query, Different Perspectives")
    print("=" * 80)
    
    # Test the same query with different personas
    test_query = "What are the latest developments in artificial intelligence?"
    print(f"Query: {test_query}")
    print("=" * 80)
    
    # Test different personas
    personas = ["Machine Learning Expert", "Business Analyst", "Curious Student", "Skeptical Reviewer"]
    
    for persona in personas:
        print(f"\n🧠 {persona}")
        print("-" * 40)
        
        try:
            agent = PersonaAgent(persona)
            result = await agent.process_query(test_query)
            
            if result["success"]:
                print("✅ Success!")
                print(f"📊 A2A Interactions: {result['a2a_interactions']}")
                if result.get("thread_id"):
                    print(f"🔗 Thread ID: {result['thread_id']}")
                
                # Show a snippet of the response
                response = result['response']
                if len(response) > 200:
                    print(f"📝 Response Preview: {response[:200]}...")
                else:
                    print(f"📝 Response: {response}")
            else:
                print(f"❌ Failed: {result['error']}")
                
        except Exception as e:
            print(f"❌ Error testing {persona}: {e}")
        
        print("-" * 40)


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
    
    # Run individual persona tests
    await test_ml_expert_persona()
    await test_business_analyst_persona()
    await test_curious_student_persona()
    await test_skeptical_reviewer_persona()
    
    # Run persona comparison test
    await test_persona_comparison()
    
    print("\n🎉 All persona tests completed!")
    print("\nThis demonstrates how different agent personas can:")
    print("1. Interact with the same A2A protocol implementation")
    print("2. Have different goals and communication styles")
    print("3. Generate different types of queries and follow-ups")
    print("4. Evaluate responses from their unique perspectives")


if __name__ == "__main__":
    asyncio.run(main()) 