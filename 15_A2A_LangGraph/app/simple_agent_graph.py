"""
Simple Agent Graph that interacts with the A2A protocol implementation.

This module creates a LangGraph-based agent that can make API calls to the existing
Agent Node through the A2A protocol, demonstrating agent-to-agent communication.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, TypedDict, Annotated
from typing_extensions import TypedDict

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleAgentState(TypedDict):
    """State for the Simple Agent Graph."""
    messages: Annotated[List, "messages"]
    a2a_responses: Annotated[List, "a2a_responses"]
    current_query: Annotated[str, "current_query"]
    agent_instructions: Annotated[str, "agent_instructions"]


class A2AClient:
    """Client for interacting with the A2A protocol server."""
    
    def __init__(self, base_url: str = "http://localhost:10000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def create_task(self, query: str) -> Dict[str, Any]:
        """Create a new task with the A2A server using JSON-RPC."""
        try:
            # Use JSON-RPC format with message/send method
            import uuid
            message_id = str(uuid.uuid4())
            
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "assistant_id": "agent",
                    "message": {
                        "messageId": message_id,
                        "role": "user",
                        "parts": [{
                            "type": "text",
                            "text": query
                        }]
                    }
                },
                "id": 1
            }
            
            response = await self.client.post(
                f"{self.base_url}/",
                json=payload,
                timeout=30.0
            )
            
            # Log the response details for debugging
            logger.info(f"A2A Response Status: {response.status_code}")
            logger.info(f"A2A Response Headers: {response.headers}")
            
            try:
                response.raise_for_status()
            except Exception as status_error:
                logger.error(f"HTTP Status Error: {status_error}")
                logger.error(f"Response Content: {response.text}")
                raise status_error
            
            task_data = response.json()
            logger.info(f"A2A Response JSON: {task_data}")
            
            # Extract both task ID and response content from JSON-RPC response
            if "result" in task_data:
                result = task_data["result"]
                task_id = result.get("id")
                
                # Extract response content from artifacts
                response_content = ""
                if "artifacts" in result:
                    for artifact in result["artifacts"]:
                        if artifact.get("name") == "result" and "parts" in artifact:
                            for part in artifact["parts"]:
                                if part.get("kind") == "text":
                                    response_content = part.get("text", "")
                                    break
                
                return {
                    "task_id": task_id,
                    "response_content": response_content,
                    "full_response": result
                }
            elif "error" in task_data:
                logger.error(f"A2A server error: {task_data['error']}")
                return None
            else:
                logger.error(f"Unexpected response format: {task_data}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating task: {e}")
            # Log additional context for debugging
            logger.error(f"Request payload: {payload}")
            logger.error(f"Base URL: {self.base_url}")
            return None
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the current status of a task using JSON-RPC."""
        # Since the A2A server returns the response immediately when creating a task,
        # we don't need to poll for status. The task_id is returned in the response.
        # For now, return a simple status indicating the task was created.
        return {
            "id": task_id,
            "status": "completed",
            "message": "Task was created and completed successfully"
        }
    
    async def add_message(self, task_id: str, message: str) -> bool:
        """Add a message to an existing task using JSON-RPC."""
        try:
            # Use JSON-RPC format with message/send method
            import uuid
            message_id = str(uuid.uuid4())
            
            payload = {
                "jsonrpc": "2.0",
                "method": "message/send",
                "params": {
                    "assistant_id": "agent",
                    "message": {
                        "messageId": message_id,
                        "role": "user",
                        "parts": [{
                            "type": "text",
                            "text": message
                        }]
                    },
                    "task_id": task_id
                },
                "id": 1
            }
            
            response = await self.client.post(
                f"{self.base_url}/",
                json=payload,
                timeout=30.0
            )
            response.raise_for_status()
            
            task_data = response.json()
            # Check for success in JSON-RPC response
            if "result" in task_data:
                return True
            elif "error" in task_data:
                logger.error(f"A2A server error: {task_data['error']}")
                return False
            else:
                logger.error(f"Unexpected response format: {task_data}")
                return False
                
        except Exception as e:
            logger.error(f"Error adding message: {e}")
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


class SimpleAgent:
    """Simple Agent that interacts with the A2A protocol implementation."""
    
    def __init__(self, a2a_url: str = "http://localhost:10000"):
        self.a2a_client = A2AClient(a2a_url)
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for the Simple Agent."""
        
        # Create the state graph
        workflow = StateGraph(SimpleAgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query_node)
        workflow.add_node("interact_with_a2a", self._interact_with_a2a_node)
        workflow.add_node("evaluate_response", self._evaluate_response_node)
        workflow.add_node("generate_final_response", self._generate_final_response_node)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add edges
        workflow.add_edge("analyze_query", "interact_with_a2a")
        workflow.add_edge("interact_with_a2a", "evaluate_response")
        workflow.add_edge("evaluate_response", "generate_final_response")
        workflow.add_edge("generate_final_response", END)
        
        # Compile the graph
        return workflow.compile(checkpointer=self.memory)
    
    async def _analyze_query_node(self, state: SimpleAgentState) -> SimpleAgentState:
        """Analyze the user query and determine how to interact with A2A."""
        messages = state["messages"]
        current_query = state["current_query"]
        
        # Create system message for analysis
        system_prompt = """You are a Simple Agent that needs to analyze user queries and determine the best way to interact with an A2A protocol server.

Your role is to:
1. Understand what the user is asking for
2. Determine if you need to use the A2A server (which has access to web search, academic papers, and document retrieval)
3. Provide clear instructions for the A2A agent

Available A2A capabilities:
- Web search for current information
- Academic paper search on arXiv
- Document retrieval from loaded documents
- Multi-turn conversations with helpfulness evaluation

Respond with clear instructions for the A2A agent."""
        
        analysis_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User query: {current_query}\n\nProvide clear instructions for the A2A agent:")
        ]
        
        response = await self.model.ainvoke(analysis_messages)
        instructions = response.content
        
        # Update state with analysis
        state["agent_instructions"] = instructions
        state["messages"].append(AIMessage(content=f"Analysis: {instructions}"))
        
        return state
    
    async def _interact_with_a2a_node(self, state: SimpleAgentState) -> SimpleAgentState:
        """Interact with the A2A protocol server."""
        current_query = state["current_query"]
        agent_instructions = state["agent_instructions"]
        
        # Combine user query with agent instructions
        enhanced_query = f"{current_query}\n\nAgent Instructions: {agent_instructions}"
        
        try:
            # Create a task with the A2A server
            task_result = await self.a2a_client.create_task(enhanced_query)
            
            if not task_result:
                error_msg = "Failed to create task with A2A server"
                state["a2a_responses"].append({"error": error_msg})
                state["messages"].append(AIMessage(content=error_msg))
                return state
            
            # Extract task ID and response content
            task_id = task_result.get("task_id")
            response_content = task_result.get("response_content", "")
            
            if task_id and response_content:
                state["a2a_responses"].append({
                    "task_id": task_id,
                    "content": response_content,
                    "status": "completed"
                })
                state["messages"].append(AIMessage(content=f"A2A Response: {response_content}"))
            else:
                state["a2a_responses"].append({
                    "task_id": task_id,
                    "content": "No response content received",
                    "status": "completed" if task_id else "failed"
                })
                
        except Exception as e:
            error_msg = f"Error interacting with A2A server: {str(e)}"
            state["a2a_responses"].append({"error": error_msg})
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    async def _evaluate_response_node(self, state: SimpleAgentState) -> SimpleAgentState:
        """Evaluate the A2A response and determine if additional interaction is needed."""
        a2a_responses = state["a2a_responses"]
        current_query = state["current_query"]
        
        if not a2a_responses:
            state["messages"].append(AIMessage(content="No A2A responses received"))
            return state
        
        # Check if we have a completed response
        completed_response = None
        for response in a2a_responses:
            if response.get("status") == "completed" and response.get("content"):
                completed_response = response["content"]
                break
        
        if completed_response:
            # Evaluate if the response is sufficient
            evaluation_prompt = f"""Given the user's original query and the A2A agent's response, determine if the response is complete and satisfactory.

User Query: {current_query}
A2A Response: {completed_response}

Evaluate if:
1. The response directly answers the user's question
2. The response provides sufficient detail
3. The response uses appropriate tools (web search, academic papers, documents)
4. No further interaction is needed

Respond with 'COMPLETE' if the response is satisfactory, or 'NEEDS_MORE' if more interaction is needed."""
            
            evaluation_messages = [
                SystemMessage(content="You are an evaluator. Respond with only 'COMPLETE' or 'NEEDS_MORE'."),
                HumanMessage(content=evaluation_prompt)
            ]
            
            evaluation = await self.model.ainvoke(evaluation_messages)
            evaluation_result = evaluation.content.strip().upper()
            
            if evaluation_result == "NEEDS_MORE":
                # Add a follow-up message to get more information
                follow_up = "The A2A agent provided some information, but I need to ask for more details to fully address your question."
                state["messages"].append(AIMessage(content=follow_up))
                
                # Try to get more information
                if a2a_responses and a2a_responses[-1].get("task_id"):
                    task_id = a2a_responses[-1]["task_id"]
                    follow_up_query = f"Please provide more detailed information about: {current_query}"
                    await self.a2a_client.add_message(task_id, follow_up_query)
        
        return state
    
    async def _generate_final_response_node(self, state: SimpleAgentState) -> SimpleAgentState:
        """Generate the final response to the user."""
        current_query = state["current_query"]
        a2a_responses = state["a2a_responses"]
        
        # Find the best response to present to the user
        best_response = None
        for response in reversed(a2a_responses):
            if response.get("content"):
                best_response = response["content"]
                break
        
        if best_response:
            final_response = f"""I've consulted with my A2A agent to help answer your question. Here's what I found:

{best_response}

This response was generated using the A2A protocol, which provides access to web search, academic papers, and document retrieval capabilities."""
        else:
            final_response = """I attempted to get information from my A2A agent, but encountered some issues. 

The A2A agent has access to:
- Web search for current information
- Academic paper search on arXiv  
- Document retrieval from loaded documents

Please try rephrasing your question or let me know if you need help with a specific type of information."""
        
        state["messages"].append(AIMessage(content=final_response))
        return state
    
    async def process_query(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """Process a user query using the Simple Agent Graph."""
        try:
            # Generate a default thread_id if none provided
            if not thread_id:
                import uuid
                thread_id = str(uuid.uuid4())
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "a2a_responses": [],
                "current_query": query,
                "agent_instructions": ""
            }
            
            # Configure the graph execution with thread_id
            config = {"configurable": {"thread_id": thread_id}}
            
            # Execute the graph
            result = await self.graph.ainvoke(initial_state, config)
            
            # Extract the final response
            final_message = result["messages"][-1]
            if isinstance(final_message, AIMessage):
                return {
                    "success": True,
                    "response": final_message.content,
                    "thread_id": thread_id,
                    "a2a_interactions": len(result["a2a_responses"])
                }
            else:
                return {
                    "success": False,
                    "error": "No final response generated",
                    "thread_id": thread_id
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "thread_id": thread_id
            }
    
    async def close(self):
        """Clean up resources."""
        await self.a2a_client.close()


# Example usage and testing
async def main():
    """Example usage of the Simple Agent."""
    agent = SimpleAgent()
    
    try:
        # Test queries
        test_queries = [
            "What are the latest developments in AI in 2024?",
            "Find recent papers on transformer architectures",
            "What do the policy documents say about student loans?"
        ]
        
        for query in test_queries:
            print(f"\n{'='*50}")
            print(f"Query: {query}")
            print(f"{'='*50}")
            
            result = await agent.process_query(query)
            
            if result["success"]:
                print(f"Response: {result['response']}")
                print(f"A2A Interactions: {result['a2a_interactions']}")
            else:
                print(f"Error: {result['error']}")
            
            print(f"{'='*50}")
            
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main()) 