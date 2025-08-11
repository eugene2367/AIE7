#!/usr/bin/env python3
"""
Simple test script to demonstrate the LangGraph workflow
"""

import asyncio
from langgraph_mcp_integration import run_workflow

async def test_workflow():
    """Test the workflow with various inputs"""
    
    test_cases = [
        "translate hello world to russian",
        "roll 2d6",
        "search for python tutorials",
        "how do you say good morning in spanish"
    ]
    
    print("🧪 Testing LangGraph MCP Workflow")
    print("=" * 50)
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_input}")
        print("-" * 40)
        
        try:
            result = await run_workflow(test_input)
            print("✅ Result:")
            print(result["final_output"])
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 40)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_workflow()) 