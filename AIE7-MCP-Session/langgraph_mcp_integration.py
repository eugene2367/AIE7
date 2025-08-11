#!/usr/bin/env python3
"""
Enhanced LangGraph Application with Real MCP Server Integration
This connects directly to your MCP server tools for real functionality
"""

import asyncio
import json
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
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
    extracted_params: Annotated[dict, "Extracted parameters from user input"]

# Real MCP Tool Integration
class RealMCPTools:
    """Real integration with your MCP server tools"""
    
    def __init__(self):
        # Import your actual MCP tools
        try:
            from server import translate_text, roll_dice, web_search
            self.translate_text = translate_text
            self.roll_dice = roll_dice
            self.web_search = web_search
            self.mcp_available = True
        except ImportError:
            print("⚠️  Warning: MCP tools not available, using fallback")
            self.mcp_available = False
    
    def translate_text(self, text: str, target_language: str = "en", source_language: str = "auto") -> str:
        """Translate text using real MCP translate_text tool"""
        if self.mcp_available:
            return self.translate_text(text, target_language, source_language)
        else:
            return f"[MCP UNAVAILABLE: Would translate '{text}' to {target_language}]"
    
    def roll_dice(self, notation: str, num_rolls: int = 1) -> str:
        """Roll dice using real MCP roll_dice tool"""
        if self.mcp_available:
            return self.roll_dice(notation, num_rolls)
        else:
            return f"[MCP UNAVAILABLE: Would roll {num_rolls}x {notation}]"
    
    def web_search(self, query: str) -> str:
        """Search web using real MCP web_search tool"""
        if self.mcp_available:
            return self.web_search(query)
        else:
            return f"[MCP UNAVAILABLE: Would search for '{query}']"

# Enhanced Parameter Extraction
def extract_translation_params(text: str) -> dict:
    """Extract translation parameters from user input"""
    text_lower = text.lower()
    
    # Common translation patterns
    patterns = [
        r"translate\s+(.+?)\s+to\s+(\w+)",
        r"translate\s+(.+?)\s+into\s+(\w+)",
        r"how\s+do\s+you\s+say\s+(.+?)\s+in\s+(\w+)",
        r"what\s+is\s+(.+?)\s+in\s+(\w+)"
    ]
    
    import re
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            text_to_translate = match.group(1).strip()
            target_lang = match.group(2).strip()
            return {
                "text": text_to_translate,
                "target_language": target_lang,
                "source_language": "auto"
            }
    
    # Fallback: simple "to" extraction
    if " to " in text_lower:
        parts = text_lower.split(" to ")
        if len(parts) >= 2:
            return {
                "text": parts[0].replace("translate", "").strip(),
                "target_language": parts[1].strip(),
                "source_language": "auto"
            }
    
    return {}

def extract_dice_params(text: str) -> dict:
    """Extract dice rolling parameters from user input"""
    text_lower = text.lower()
    
    # Common dice patterns
    patterns = [
        r"roll\s+(\d+d\d+)(?:\s+(\d+)\s+times?)?",
        r"(\d+d\d+)(?:\s+(\d+)\s+times?)?",
        r"dice\s+(\d+d\d+)(?:\s+(\d+)\s+times?)?"
    ]
    
    import re
    
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            notation = match.group(1)
            num_rolls = int(match.group(2)) if match.group(2) else 1
            return {
                "notation": notation,
                "num_rolls": num_rolls
            }
    
    return {}

def extract_search_params(text: str) -> dict:
    """Extract web search parameters from user input"""
    text_lower = text.lower()
    
    # Common search patterns
    search_terms = ["search for", "find", "look up", "search", "find information about"]
    
    for term in search_terms:
        if term in text_lower:
            query = text_lower.replace(term, "").strip()
            return {"query": query}
    
    return {}

# Enhanced Workflow Nodes
def analyze_request(state: WorkflowState) -> WorkflowState:
    """Analyze the user request and determine the task with parameter extraction"""
    messages = state["messages"]
    last_message = messages[-1] if messages else ""
    
    # Task classification with confidence
    task_scores = {
        "translation": 0,
        "dice_rolling": 0,
        "web_search": 0
    }
    
    # Score based on keywords
    translation_keywords = ["translate", "translation", "language", "say", "word"]
    dice_keywords = ["dice", "roll", "random", "d20", "d6", "d100"]
    search_keywords = ["search", "find", "look up", "information", "about"]
    
    for keyword in translation_keywords:
        if keyword in last_message.lower():
            task_scores["translation"] += 1
    
    for keyword in dice_keywords:
        if keyword in last_message.lower():
            task_scores["dice_rolling"] += 1
    
    for keyword in search_keywords:
        if keyword in last_message.lower():
            task_scores["web_search"] += 1
    
    # Determine the task
    if max(task_scores.values()) > 0:
        current_task = max(task_scores, key=task_scores.get)
    else:
        current_task = "general_assistance"
    
    # Extract parameters based on task
    extracted_params = {}
    if current_task == "translation":
        extracted_params = extract_translation_params(last_message)
    elif current_task == "dice_rolling":
        extracted_params = extract_dice_params(last_message)
    elif current_task == "web_search":
        extracted_params = extract_search_params(last_message)
    
    return {
        **state,
        "current_task": current_task,
        "extracted_params": extracted_params,
        "next_action": "execute_task"
    }

def execute_task(state: WorkflowState) -> WorkflowState:
    """Execute the identified task using real MCP tools"""
    current_task = state["current_task"]
    extracted_params = state["extracted_params"]
    messages = state["messages"]
    last_message = messages[-1] if messages else ""
    
    # Initialize MCP tools
    mcp_tools = RealMCPTools()
    
    if current_task == "translation":
        if extracted_params:
            text = extracted_params["text"]
            target_lang = extracted_params["target_language"]
            source_lang = extracted_params.get("source_language", "auto")
            
            try:
                result = mcp_tools.translate_text(text, target_lang, source_lang)
            except Exception as e:
                result = f"❌ Translation error: {str(e)}"
        else:
            result = "Please specify what to translate and to which language (e.g., 'translate hello world to russian')"
        
        next_action = "format_output"
        
    elif current_task == "dice_rolling":
        if extracted_params:
            notation = extracted_params["notation"]
            num_rolls = extracted_params["num_rolls"]
            
            try:
                result = mcp_tools.roll_dice(notation, num_rolls)
            except Exception as e:
                result = f"❌ Dice rolling error: {str(e)}"
        else:
            result = "Please specify dice notation (e.g., 'roll 2d6' or '2d20 3 times')"
        
        next_action = "format_output"
        
    elif current_task == "web_search":
        if extracted_params:
            query = extracted_params["query"]
            
            try:
                result = mcp_tools.web_search(query)
            except Exception as e:
                result = f"❌ Web search error: {str(e)}"
        else:
            result = "Please specify what to search for (e.g., 'search for python tutorials')"
        
        next_action = "format_output"
        
    else:
        result = """🤖 I can help with:

🌍 Translation: "translate hello world to russian"
🎲 Dice rolling: "roll 2d6" or "2d20 3 times"  
🔍 Web search: "search for python tutorials"

What would you like to do?"""
        next_action = "format_output"
    
    return {
        **state,
        "task_result": result,
        "next_action": next_action
    }

def format_output(state: WorkflowState) -> WorkflowState:
    """Format the final output with enhanced formatting"""
    task_result = state["task_result"]
    current_task = state["current_task"]
    extracted_params = state["extracted_params"]
    
    # Enhanced formatting based on task type
    if current_task == "translation":
        if extracted_params:
            final_output = f"""🌍 Translation Complete!

📤 Original: "{extracted_params['text']}"
📥 Target Language: {extracted_params['target_language'].upper()}

{task_result}"""
        else:
            final_output = task_result
            
    elif current_task == "dice_rolling":
        if extracted_params:
            final_output = f"""🎲 Dice Roll Complete!

🎯 Notation: {extracted_params['notation']}
🔄 Rolls: {extracted_params['num_rolls']}

{task_result}"""
        else:
            final_output = task_result
            
    elif current_task == "web_search":
        if extracted_params:
            final_output = f"""🔍 Web Search Complete!

🔎 Query: "{extracted_params['query']}"

{task_result}"""
        else:
            final_output = task_result
    else:
        final_output = task_result
    
    return {
        **state,
        "final_output": final_output,
        "next_action": "end"
    }

def should_continue(state: WorkflowState) -> str:
    """Determine if we should continue or end"""
    next_action = state.get("next_action", "")
    return "end" if next_action == "end" else "format_output"

# Build the workflow
def build_workflow():
    """Build the enhanced LangGraph workflow"""
    
    # Create the workflow
    workflow = StateGraph(WorkflowState)
    
    # Add nodes
    workflow.add_node("analyze_request", analyze_request)
    workflow.add_node("execute_task", execute_task)
    workflow.add_node("format_output", format_output)
    
    # Add edges - simplified linear flow
    workflow.add_edge("analyze_request", "execute_task")
    workflow.add_edge("execute_task", "format_output")
    workflow.add_edge("format_output", END)
    
    # Set entry point
    workflow.set_entry_point("analyze_request")
    
    return workflow.compile()

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
        "final_output": "",
        "extracted_params": {}
    }
    
    # Run the workflow
    result = await workflow.ainvoke(initial_state)
    
    return result

# CLI interface for testing
async def main():
    """Main function for testing the enhanced workflow"""
    print("🚀 Enhanced LangGraph MCP Integration Demo")
    print("=" * 50)
    print("This app connects to your real MCP server tools!")
    print("=" * 50)
    print("Available tasks:")
    print("🌍 Translation: 'translate hello world to russian'")
    print("🎲 Dice rolling: 'roll 2d6' or '2d20 3 times'")
    print("🔍 Web search: 'search for python tutorials'")
    print("💡 Type 'quit' to exit")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🤖 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("\n🔄 Processing with LangGraph workflow...")
            result = await run_workflow(user_input)
            
            print(f"\n📤 Result:")
            print(result["final_output"])
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 