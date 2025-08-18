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
    
    agent = PersonaAgent("ml_expert")
    
    try:
        # Test query that would interest an ML expert
        test_query = "What makes Kimi K2 so incredible? I want technical details and academic sources."
        
        print(f"Query: {test_query}")
        print(f"Persona Goals: {', '.join(agent.persona_config['goals'])}")
        print(f"Persona Style: {agent.persona_config['style']}")
        print("-" * 60)
        
        result = await agent.process_query(test_query)
        
        if result["success"]:
            print("✅ Success!")
            print(f"Response: {result['response']}")
            print(f"A2A Interactions: {result['a2a_interactions']}")
        else:
            print("❌ Failed!")
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await agent.close()


async def test_business_analyst_persona():
    """Test the Business Analyst persona."""
    
    print("\n💼 Testing Business Analyst Persona")
    print("=" * 60)
    
    agent = PersonaAgent("business_analyst")
    
    try:
        # Test query that would interest a business analyst
        test_query = "What are the business implications of recent AI developments? I need practical insights and market data."
        
        print(f"Query: {test_query}")
        print(f"Persona Goals: {', '.join(agent.persona_config['goals'])}")
        print(f"Persona Style: {agent.persona_config['style']}")
        print("-" * 60)
        
        result = await agent.process_query(test_query)
        
        if result["success"]:
            print("✅ Success!")
            print(f"Response: {result['response']}")
            print(f"A2A Interactions: {result['a2a_interactions']}")
        else:
            print("❌ Failed!")
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        await agent.close()


async def test_curious_student_persona():
    """Test the Curious Student persona."""
    
    print("\n🎓 Testing Curious Student Persona")
    print("=" * 60)
    
    agent = PersonaAgent("curious_student")
    
    try:
        # Test query that would interest a curious student
        test_query = "Can you explain what transformers are in simple terms? I want to understand the basics and see real examples."
        
        print(f"Query: {test_query}")
        print(f"Persona Goals: {', '.join(agent.persona_config['goals'])}")
        print(f"Persona Style: {agent.persona_config['style']}")
        print("-" * 60)
        
        result = await agent.process_query(test_query)
        
        if result["success"]:
            print("✅ Success!")
            print(f"Response: {result['response']}")
            print(f"A2A Interactions: {result['a2a_interactions']}")
        else:
            print("❌ Failed!")
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        await agent.close()


async def test_skeptical_reviewer_persona():
    """Test the Skeptical Reviewer persona."""
    
    print("\n🔍 Testing Skeptical Reviewer Persona")
    print("=" * 60)
    
    agent = PersonaAgent("skeptical_reviewer")
    
    try:
        # Test query that would interest a skeptical reviewer
        test_query = "What evidence supports the claims about AI capabilities? I want to see multiple sources and understand limitations."
        
        print(f"Query: {test_query}")
        print(f"Persona Goals: {', '.join(agent.persona_config['goals'])}")
        print(f"Persona Style: {agent.persona_config['style']}")
        print("-" * 60)
        
        result = await agent.process_query(test_query)
        
        if result["success"]:
            print("✅ Success!")
            print(f"Response: {result['response']}")
            print(f"A2A Interactions: {result['a2a_interactions']}")
        else:
            print("❌ Failed!")
            print(f"Error: {result['error']}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
    
    finally:
        await agent.close()


async def test_persona_comparison():
    """Test all personas with the same query to see how they differ."""
    
    print("\n🔄 Testing Persona Comparison - Same Query, Different Perspectives")
    print("=" * 80)
    
    personas = ["ml_expert", "business_analyst", "curious_student", "skeptical_reviewer"]
    test_query = "What are the latest developments in artificial intelligence?"
    
    print(f"Query: {test_query}")
    print("=" * 80)
    
    results = {}
    
    for persona in personas:
        print(f"\n🧠 {PersonaAgent.PERSONAS[persona]['name']}")
        print("-" * 40)
        
        agent = PersonaAgent(persona)
        
        try:
            result = await agent.process_query(test_query)
            
            if result["success"]:
                print("✅ Success!")
                print(f"Response Preview: {result['response'][:200]}...")
                print(f"A2A Interactions: {result['a2a_interactions']}")
                results[persona] = result
            else:
                print("❌ Failed!")
                print(f"Error: {result['error']}")
        
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        finally:
            await agent.close()
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("📊 PERSONA COMPARISON SUMMARY")
    print("=" * 80)
    
    for persona, result in results.items():
        if result and result.get("success"):
            print(f"\n{PersonaAgent.PERSONAS[persona]['name']}:")
            print(f"  - A2A Interactions: {result['a2a_interactions']}")
            print(f"  - Response Length: {len(result['response'])} characters")
            print(f"  - Style: {PersonaAgent.PERSONAS[persona]['style']}")


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