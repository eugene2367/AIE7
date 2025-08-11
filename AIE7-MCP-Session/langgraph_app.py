#!/usr/bin/env python3
"""
LangGraph Application that interacts with MCP Server Tools
This creates a workflow that can use translation, dice rolling, and web search capabilities
"""

import asyncio
import json
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# State definition for our workflow
class WorkflowState(TypedDict):
    """State for the workflow"""
    messages: Annotated[Sequence[str], "List of messages in the conversation"]
    current_task: Annotated[str, "Current task being processed"]
    task_result: Annotated[str, "Result of the current task"]
    next_action: Annotated[str, "Next action to take"]
    final_output: Annotated[str, "Final output of the workflow"]

# MCP Tool Wrappers (these would normally connect to your MCP server)
class MCPToolWrapper:
    """Wrapper for MCP tools - in production this would connect to your MCP server"""
    
    @staticmethod
    def translate_text(text: str, target_language: str = "en", source_language: str = "auto") -> str:
        """Translate text using MCP translate_text tool"""
        # This is a mock - in production you'd call your actual MCP server
        translations = {
            "ru": {"hello world": "Привет, мир", "good morning": "Доброе утро"},
            "es": {"hello world": "Hola mundo", "good morning": "Buenos días"},
            "fr": {"hello world": "Bonjour le monde", "good morning": "Bonjour"},
            "de": {"hello world": "Hallo Welt", "good morning": "Guten Morgen"}
        }
        
        if target_language in translations and text.lower() in translations[target_language]:
            return translations[target_language][text.lower()]
        return f"[TRANSLATED: {text} to {target_language}]"
    
    @staticmethod
    def roll_dice(notation: str, num_rolls: int = 1) -> str:
        """Roll dice using MCP roll_dice tool"""
        # Mock dice rolling - in production you'd call your actual MCP server
        return f"[DICE ROLL: {num_rolls}x {notation} = Random result]"
    
    @staticmethod
    def web_search(query: str) -> str:
        """Search web using MCP web_search tool"""
        # Mock web search - in production you'd call your actual MCP server
        return f"[WEB SEARCH: {query} - Found relevant information]"

# Workflow Nodes
def analyze_request(state: WorkflowState) -> WorkflowState:
    """Analyze the user request and determine the task"""
    messages = state["messages"]
    last_message = messages[-1] if messages else ""
    
    # Simple task classification
    if any(word in last_message.lower() for word in ["translate", "translation", "language"]):
        task = "translation"
    elif any(word in last_message.lower() for word in ["dice", "roll", "random"]):
        task = "dice_rolling"
    elif any(word in last_message.lower() for word in ["search", "find", "web"]):
        task = "web_search"
    else:
        task = "general_assistance"
    
    return {
        **state,
        "current_task": task,
        "next_action": "execute_task"
    }

def execute_task(state: WorkflowState) -> WorkflowState:
    """Execute the identified task using MCP tools"""
    current_task = state["current_task"]
    messages = state["messages"]
    last_message = messages[-1] if messages else ""
    
    mcp_tools = MCPToolWrapper()
    
    if current_task == "translation":
        # Extract translation parameters
        if "to" in last_message.lower():
            parts = last_message.lower().split("to")
            text_to_translate = parts[0].strip()
            target_lang = parts[1].strip()
            result = mcp_tools.translate_text(text_to_translate, target_lang)
        else:
            result = "Please specify what to translate and to which language (e.g., 'translate hello world to russian')"
        
        next_action = "format_output"
        
    elif current_task == "dice_rolling":
        # Extract dice notation
        if "d" in last_message.lower():
            # Simple dice notation extraction
            result = mcp_tools.roll_dice("2d6", 1)
        else:
            result = "Please specify dice notation (e.g., 'roll 2d6')"
        
        next_action = "format_output"
        
    elif current_task == "web_search":
        # Extract search query
        search_terms = ["search for", "find", "look up"]
        query = last_message
        for term in search_terms:
            if term in last_message.lower():
                query = last_message.lower().replace(term, "").strip()
                break
        
        result = mcp_tools.web_search(query)
        next_action = "format_output"
        
    else:
        result = "I can help with translation, dice rolling, and web search. What would you like to do?"
        next_action = "format_output"
    
    return {
        **state,
        "task_result": result,
        "next_action": next_action
    }

def format_output(state: WorkflowState) -> WorkflowState:
    """Format the final output"""
    task_result = state["task_result"]
    current_task = state["current_task"]
    
    # Format based on task type
    if current_task == "translation":
        final_output = f"🌍 Translation Complete!\n\n{task_result}"
    elif current_task == "dice_rolling":
        final_output = f"🎲 Dice Roll Complete!\n\n{task_result}"
    elif current_task == "web_search":
        final_output = f"🔍 Web Search Complete!\n\n{task_result}"
    else:
        final_output = task_result
    
    return {
        **state,
        "final_output": final_output,
        "next_action": "end"
    }

def should_continue(state: WorkflowState) -> str:
    """Determine if we should continue or end"""
    return "end" if state["next_action"] == "end" else "execute_task"

# Build the workflow
def build_workflow():
    """Build the LangGraph workflow"""
    
    # Create the workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyze_request", analyze_request)
    workflow.add_node("execute_task", execute_task)
    workflow.add_node("format_output", format_output)
    
    # Add edges
    workflow.add_edge("analyze_request", "execute_task")
    workflow.add_conditional_edges(
        "execute_task",
        should_continue,
        {
            "format_output": "format_output",
            "end": END
        }
    )
    workflow.add_edge("format_output", END)
    
    # Set entry point
    workflow.set_entry_point("analyze_request")
    
    return workflow.compile(checkpointer=MemorySaver())

# Main execution function
async def run_workflow(user_input: str):
    """Run the workflow with user input"""
    workflow = build_workflow()
    
    # Initialize state
    initial_state = {
        "messages": [user_input],
        "current_task": "",
        "task_result": "",
        "next_action": "",
        "final_output": ""
    }
    
    # Run the workflow
    result = await workflow.ainvoke(initial_state)
    
    return result

# CLI interface for testing
async def main():
    """Main function for testing the workflow"""
    print("🌍 LangGraph MCP Workflow Demo")
    print("=" * 40)
    print("Available tasks:")
    print("- Translation: 'translate hello world to russian'")
    print("- Dice rolling: 'roll 2d6'")
    print("- Web search: 'search for python tutorials'")
    print("- Type 'quit' to exit")
    print("=" * 40)
    
    while True:
        try:
            user_input = input("\n🤖 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🔄 Processing...")
            result = await run_workflow(user_input)
            
            print(f"\n📤 Result:")
            print(result["final_output"])
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 