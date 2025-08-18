"""
Comprehensive Demo of the A2A Protocol Workflow with LangGraph Agents.

This script demonstrates the complete workflow:
1. Simple Agent interacting with A2A protocol
2. Persona Agents with different goals and styles
3. Comparison of different approaches
4. Real-time interaction examples
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path to import app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.simple_agent_graph import SimpleAgent
from app.persona_agent_graph import PersonaAgent


class A2AWorkflowDemo:
    """Comprehensive demo of the A2A protocol workflow."""
    
    def __init__(self):
        self.simple_agent = None
        self.persona_agents = {}
        self.demo_results = {}
    
    async def setup_agents(self):
        """Initialize all agents."""
        print("🚀 Setting up A2A Protocol Agents...")
        
        # Initialize Simple Agent
        self.simple_agent = SimpleAgent()
        
        # Initialize Persona Agents
        personas = ["ml_expert", "business_analyst", "curious_student", "skeptical_reviewer"]
        for persona in personas:
            self.persona_agents[persona] = PersonaAgent(persona)
        
        print("✅ All agents initialized successfully!")
    
    async def demo_simple_agent(self):
        """Demonstrate the Simple Agent workflow."""
        print("\n🤖 DEMO 1: Simple Agent - Basic A2A Interaction")
        print("=" * 70)
        
        if not self.simple_agent:
            print("❌ Simple Agent not initialized")
            return
        
        # Test queries for different tool types
        test_queries = [
            {
                "query": "What are the latest developments in AI in 2024?",
                "expected_tool": "Web Search",
                "description": "Current information query"
            },
            {
                "query": "Find recent papers on transformer architectures",
                "expected_tool": "Academic Search",
                "description": "Research paper query"
            },
            {
                "query": "What do the policy documents say about requirements?",
                "expected_tool": "Document Retrieval",
                "description": "Document search query"
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            print(f"Expected Tool: {test_case['expected_tool']}")
            print("-" * 50)
            
            try:
                result = await self.simple_agent.process_query(test_case['query'])
                
                if result["success"]:
                    print("✅ Success!")
                    print(f"Response Preview: {result['response'][:150]}...")
                    print(f"A2A Interactions: {result['a2a_interactions']}")
                    results.append({
                        "test_case": test_case,
                        "result": result,
                        "status": "success"
                    })
                else:
                    print("❌ Failed!")
                    print(f"Error: {result['error']}")
                    results.append({
                        "test_case": test_case,
                        "result": result,
                        "status": "failed"
                    })
                
            except Exception as e:
                print(f"❌ Exception: {e}")
                results.append({
                    "test_case": test_case,
                    "error": str(e),
                    "status": "exception"
                })
        
        self.demo_results["simple_agent"] = results
        print(f"\n📊 Simple Agent Demo Results: {len([r for r in results if r['status'] == 'success'])}/{len(results)} successful")
    
    async def demo_persona_agents(self):
        """Demonstrate the Persona Agents workflow."""
        print("\n🧠 DEMO 2: Persona Agents - Specialized A2A Interaction")
        print("=" * 70)
        
        if not self.persona_agents:
            print("❌ Persona Agents not initialized")
            return
        
        # Test the same query with different personas
        test_query = "What makes Kimi K2 so incredible? I want to understand why it's special."
        
        print(f"Test Query: {test_query}")
        print("Testing how different personas approach the same question...")
        
        results = {}
        
        for persona_name, agent in self.persona_agents.items():
            print(f"\n🧠 {PersonaAgent.PERSONAS[persona_name]['name']}")
            print(f"Goals: {', '.join(agent.persona_config['goals'])}")
            print(f"Style: {agent.persona_config['style']}")
            print("-" * 50)
            
            try:
                result = await agent.process_query(test_query)
                
                if result["success"]:
                    print("✅ Success!")
                    print(f"Response Preview: {result['response'][:200]}...")
                    print(f"A2A Interactions: {result['a2a_interactions']}")
                    results[persona_name] = {
                        "result": result,
                        "status": "success"
                    }
                else:
                    print("❌ Failed!")
                    print(f"Error: {result['error']}")
                    results[persona_name] = {
                        "result": result,
                        "status": "failed"
                    }
                
            except Exception as e:
                print(f"❌ Exception: {e}")
                results[persona_name] = {
                    "error": str(e),
                    "status": "exception"
                }
        
        self.demo_results["persona_agents"] = results
        print(f"\n📊 Persona Agents Demo Results: {len([r for r in results.values() if r['status'] == 'success'])}/{len(results)} successful")
    
    async def demo_conversation_flow(self):
        """Demonstrate multi-turn conversation flow."""
        print("\n🔄 DEMO 3: Multi-Turn Conversation Flow")
        print("=" * 70)
        
        if not self.simple_agent:
            print("❌ Simple Agent not initialized")
            return
        
        conversation = [
            "What are the latest developments in AI?",
            "Can you provide more details about transformer models?",
            "What are the practical applications of these developments?"
        ]
        
        print("Starting a multi-turn conversation...")
        results = []
        
        for i, message in enumerate(conversation, 1):
            print(f"\n💬 Turn {i}: {message}")
            print("-" * 40)
            
            try:
                result = await self.simple_agent.process_query(message)
                
                if result["success"]:
                    print("✅ Success!")
                    print(f"Response: {result['response'][:150]}...")
                    print(f"A2A Interactions: {result['a2a_interactions']}")
                    results.append({
                        "turn": i,
                        "message": message,
                        "result": result,
                        "status": "success"
                    })
                else:
                    print("❌ Failed!")
                    print(f"Error: {result['error']}")
                    results.append({
                        "turn": i,
                        "message": message,
                        "result": result,
                        "status": "failed"
                    })
                
            except Exception as e:
                print(f"❌ Exception: {e}")
                results.append({
                    "turn": i,
                    "message": message,
                    "error": str(e),
                    "status": "exception"
                })
        
        self.demo_results["conversation_flow"] = results
        print(f"\n📊 Conversation Flow Results: {len([r for r in results if r['status'] == 'success'])}/{len(results)} successful")
    
    async def demo_persona_comparison(self):
        """Demonstrate how different personas handle the same query."""
        print("\n🔍 DEMO 4: Persona Comparison - Same Query, Different Approaches")
        print("=" * 70)
        
        if not self.persona_agents:
            print("❌ Persona Agents not initialized")
            return
        
        # Use a query that would interest all personas
        test_query = "What are the latest developments in artificial intelligence?"
        
        print(f"Test Query: {test_query}")
        print("Comparing how different personas approach this question...")
        
        results = {}
        
        for persona_name, agent in self.persona_agents.items():
            print(f"\n🧠 {PersonaAgent.PERSONAS[persona_name]['name']}")
            print("-" * 40)
            
            try:
                result = await agent.process_query(test_query)
                
                if result["success"]:
                    print("✅ Success!")
                    print(f"Response Length: {len(result['response'])} characters")
                    print(f"A2A Interactions: {result['a2a_interactions']}")
                    print(f"Response Preview: {result['response'][:150]}...")
                    results[persona_name] = {
                        "result": result,
                        "status": "success"
                    }
                else:
                    print("❌ Failed!")
                    print(f"Error: {result['error']}")
                    results[persona_name] = {
                        "result": result,
                        "status": "failed"
                    }
                
            except Exception as e:
                print(f"❌ Exception: {e}")
                results[persona_name] = {
                    "error": str(e),
                    "status": "exception"
                }
        
        self.demo_results["persona_comparison"] = results
        
        # Summary comparison
        print("\n" + "=" * 70)
        print("📊 PERSONA COMPARISON SUMMARY")
        print("=" * 70)
        
        for persona_name, result in results.items():
            if result and result.get("status") == "success":
                persona_info = PersonaAgent.PERSONAS[persona_name]
                print(f"\n{persona_info['name']}:")
                print(f"  - Style: {persona_info['style']}")
                print(f"  - A2A Interactions: {result['result']['a2a_interactions']}")
                print(f"  - Response Length: {len(result['result']['response'])} characters")
                print(f"  - Key Goals: {', '.join(persona_info['goals'][:2])}...")
    
    def generate_demo_report(self):
        """Generate a comprehensive demo report."""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE DEMO REPORT")
        print("=" * 80)
        
        total_tests = 0
        successful_tests = 0
        
        for demo_name, results in self.demo_results.items():
            if isinstance(results, list):
                # Simple agent or conversation flow results
                demo_total = len(results)
                demo_success = len([r for r in results if r.get('status') == 'success'])
            else:
                # Persona agent results
                demo_total = len(results)
                demo_success = len([r for r in results.values() if r.get('status') == 'success'])
            
            total_tests += demo_total
            successful_tests += demo_success
            
            print(f"\n{demo_name.replace('_', ' ').title()}:")
            print(f"  - Tests: {demo_success}/{demo_total} successful")
            print(f"  - Success Rate: {(demo_success/demo_total*100):.1f}%")
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"  - Total Tests: {total_tests}")
        print(f"  - Successful: {successful_tests}")
        print(f"  - Overall Success Rate: {(successful_tests/total_tests*100):.1f}%")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print("1. Simple Agent provides basic A2A protocol interaction")
        print("2. Persona Agents demonstrate specialized communication styles")
        print("3. Different personas generate different types of queries")
        print("4. A2A protocol enables consistent agent-to-agent communication")
        print("5. LangGraph provides structured workflow management")
    
    async def run_complete_demo(self):
        """Run the complete A2A workflow demo."""
        print("🚀 A2A Protocol Workflow Demo with LangGraph Agents")
        print("=" * 80)
        
        try:
            # Setup all agents
            await self.setup_agents()
            
            # Run all demos
            await self.demo_simple_agent()
            await self.demo_persona_agents()
            await self.demo_conversation_flow()
            await self.demo_persona_comparison()
            
            # Generate comprehensive report
            self.generate_demo_report()
            
            print("\n🎉 Demo completed successfully!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Clean up all agents."""
        print("\n🧹 Cleaning up resources...")
        
        if self.simple_agent:
            await self.simple_agent.close()
        
        for agent in self.persona_agents.values():
            await agent.close()
        
        print("✅ Cleanup completed!")


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
    """Main demo function."""
    
    print("🔍 Checking A2A server status...")
    if not check_a2a_server():
        print("❌ A2A server is not running on http://localhost:10000")
        print("Please start the A2A server first:")
        print("  uv run python -m app")
        print("  # or")
        print("  ./quickstart.sh")
        return
    
    print("✅ A2A server is running!")
    
    # Create and run the demo
    demo = A2AWorkflowDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main()) 