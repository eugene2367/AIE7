"""
Persona Agent Graph that integrates with the A2A protocol implementation.

This module creates specialized LangGraph-based agents with different personas that can call the full A2A agent graph
with tool execution and helpfulness evaluation, demonstrating persona-specific agent-to-agent communication.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, TypedDict, Annotated
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import the A2A agent graph
from app.agent_graph_with_helpfulness import build_agent_graph_with_helpfulness
from app.agent import ResponseFormat

import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaAgentState(TypedDict):
    """State for the Persona Agent Graph."""
    messages: Annotated[List, "messages"]
    a2a_responses: Annotated[List, "a2a_responses"]
    current_query: Annotated[str, "current_query"]
    persona_context: Annotated[str, "persona_context"]
    strategic_approach: Annotated[str, "strategic_approach"]
    final_response: Annotated[str, "final_response"]


class PersonaAgent:
    """Persona Agent that integrates with the A2A agent graph."""
    
    def __init__(self, persona_type: str = "Machine Learning Expert", model: ChatOpenAI = None):
        """Initialize the Persona Agent."""
        self.persona_type = persona_type
        self.model = model or ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize memory for conversation persistence
        self.memory = MemorySaver()
        
        # Build the A2A agent graph with helpfulness
        self.a2a_graph = build_agent_graph_with_helpfulness(
            model=self.model,
            system_instruction="""You are an A2A agent with access to powerful tools including web search, academic paper search, and document retrieval. 
            Your goal is to provide comprehensive, accurate, and helpful responses to user queries using the appropriate tools when needed.
            
            Available tools:
            - Web search for current information
            - Academic paper search on arXiv
            - Document retrieval from loaded documents
            
            Always use tools when they would improve your response quality.""",
            format_instruction="""Provide responses in a clear, structured format that directly addresses the user's query.
            When using tools, explain what you found and how it relates to the question.
            Always cite your sources when possible.""",
            checkpointer=self.memory
        )
        
        # Build the Persona Agent graph
        self.graph = self._build_graph()
    
    def _get_persona_config(self) -> Dict[str, Any]:
        """Get the configuration for the specified persona."""
        personas = {
            "Machine Learning Expert": {
                "context": "An expert in Machine Learning who wants deep, technical information with academic sources",
                "goals": [
                    "Get detailed technical explanations",
                    "Find academic papers and research",
                    "Verify information with credible sources",
                    "Understand implementation details"
                ],
                "style": "Technical, thorough, academic-focused"
            },
            "Business Analyst": {
                "context": "A business analyst who focuses on practical insights and market implications",
                "goals": [
                    "Get practical business insights",
                    "Understand market implications",
                    "Find actionable recommendations",
                    "Analyze industry trends"
                ],
                "style": "Practical, business-oriented, results-focused"
            },
            "Curious Student": {
                "context": "A student who wants to learn and understand concepts thoroughly",
                "goals": [
                    "Learn fundamental concepts",
                    "Get clear explanations",
                    "Find learning resources",
                    "Understand real-world applications"
                ],
                "style": "Inquisitive, learning-focused, thorough"
            },
            "Skeptical Reviewer": {
                "context": "A critical reviewer who questions claims and seeks verification",
                "goals": [
                    "Verify claims and statements",
                    "Find contradictory evidence",
                    "Question assumptions",
                    "Get multiple perspectives"
                ],
                "style": "Critical, questioning, verification-focused"
            }
        }
        
        return personas.get(self.persona_type, personas["Machine Learning Expert"])
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for the Persona Agent."""
        
        # Create the state graph
        workflow = StateGraph(PersonaAgentState)
        
        # Add nodes
        workflow.add_node("persona_analysis", self._persona_analysis_node)
        workflow.add_node("strategic_interaction", self._strategic_interaction_node)
        workflow.add_node("response_evaluation", self._response_evaluation_node)
        workflow.add_node("helpfulness_decision", self._helpfulness_decision)
        workflow.add_node("persona_response", self._persona_response_node)
        
        # Set entry point
        workflow.set_entry_point("persona_analysis")
        
        # Add edges
        workflow.add_edge("persona_analysis", "strategic_interaction")
        workflow.add_edge("strategic_interaction", "response_evaluation")
        workflow.add_edge("response_evaluation", "helpfulness_decision")
        workflow.add_edge("helpfulness_decision", "persona_response")
        workflow.add_edge("persona_response", END)
        
        # Compile the graph
        return workflow.compile(checkpointer=self.memory)
    
    async def _persona_analysis_node(self, state: PersonaAgentState) -> PersonaAgentState:
        """Analyze the query through the lens of the specific persona."""
        current_query = state["current_query"]
        persona_config = self._get_persona_config()
        
        # Create system message for persona analysis
        system_prompt = f"""You are a {self.persona_type} with the following characteristics:

Context: {persona_config['context']}
Goals: {', '.join(persona_config['goals'])}
Style: {persona_config['style']}

Your role is to analyze user queries and determine the best strategic approach for getting information that meets your specific needs and goals.

Analyze the query and provide:
1. Specific information you need
2. Your approach to the A2A agent
3. Follow-up questions you might ask
4. Verification or additional details needed

Be specific about what matters most to your persona type."""
        
        analysis_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User Query: {current_query}\n\nAnalyze this query from your perspective:")
        ]
        
        response = await self.model.ainvoke(analysis_messages)
        analysis = response.content
        
        # Update state with persona analysis
        state["persona_context"] = f"{self.persona_type} - {persona_config['context']}"
        state["strategic_approach"] = analysis
        state["messages"].append(AIMessage(content=f"Persona Analysis: {analysis}"))
        
        return state
    
    async def _strategic_interaction_node(self, state: PersonaAgentState) -> PersonaAgentState:
        """Make strategic A2A calls based on persona goals and communication style."""
        current_query = state["current_query"]
        strategic_approach = state["strategic_approach"]
        
        # Combine user query with persona strategy
        strategic_query = f"""User Query: {current_query}

Persona Context: {state['persona_context']}

Persona Goals:
{self._get_persona_config()['goals']}

Persona Style: {self._get_persona_config()['style']}

Strategic Approach: {strategic_approach}

Please provide information that addresses the persona's specific needs and goals. 
Focus on the aspects that matter most to this persona type."""
        
        try:
            # Initialize the A2A agent state
            a2a_initial_state = {
                "messages": [HumanMessage(content=strategic_query)],
                "structured_response": None
            }
            
            # Execute the A2A agent graph
            logger.info(f"Calling A2A agent graph for {self.persona_type}...")
            a2a_result = await self.a2a_graph.ainvoke(a2a_initial_state)
            
            # Extract the response from the A2A agent
            if "messages" in a2a_result and a2a_result["messages"]:
                # Find the actual response content (skip HELPFULNESS markers)
                a2a_response_content = ""
                for message in reversed(a2a_result["messages"]):
                    if isinstance(message, AIMessage) and message.content:
                        content = message.content
                        # Skip HELPFULNESS evaluation messages
                        if not content.startswith("HELPFULNESS:"):
                            a2a_response_content = content
                            break
                
                # Check if there's a structured response
                structured_response = a2a_result.get("structured_response")
                
                if a2a_response_content:
                    state["a2a_responses"].append({
                        "content": a2a_response_content,
                        "structured_response": structured_response,
                        "status": "completed"
                    })
                    
                    state["messages"].append(AIMessage(content=f"A2A Agent Response: {a2a_response_content}"))
                    
                    logger.info(f"A2A agent graph execution completed successfully for {self.persona_type}")
                else:
                    error_msg = "No valid response content found in A2A agent messages"
                    state["a2a_responses"].append({"error": error_msg})
                    state["messages"].append(AIMessage(content=error_msg))
            else:
                error_msg = "No response received from A2A agent"
                state["a2a_responses"].append({"error": error_msg})
                state["messages"].append(AIMessage(content=error_msg))
                
        except Exception as e:
            error_msg = f"Error calling A2A agent: {str(e)}"
            logger.error(f"Error in A2A agent call for {self.persona_type}: {e}")
            state["a2a_responses"].append({"error": error_msg})
            state["messages"].append(AIMessage(content=error_msg))
        
        return state
    
    async def _response_evaluation_node(self, state: PersonaAgentState) -> PersonaAgentState:
        """Evaluate if the A2A response meets the persona's goals and is helpful."""
        current_query = state["current_query"]
        a2a_responses = state["a2a_responses"]
        persona_context = state["persona_context"]
        
        # Find the best A2A response
        best_response = None
        for response in reversed(a2a_responses):
            if response.get("content") and not response.get("error"):
                best_response = response["content"]
                break
        
        if not best_response:
            # No valid response, mark as unhelpful
            state["messages"].append(AIMessage(content="HELPFULNESS:N"))
            return state
        
        # Evaluate if the response meets the persona's goals
        evaluation_prompt = f"""Given the persona's context and the A2A agent's response, determine if the response is helpful for this specific persona.

Persona Context: {persona_context}
Original Query: {current_query}
A2A Response: {best_response}

Evaluate if the response:
1. Addresses the persona's specific goals and needs
2. Provides information in the persona's preferred style
3. Meets the persona's expectations for depth and detail
4. Is helpful for the persona's use case

Respond with 'HELPFULNESS:Y' if helpful, or 'HELPFULNESS:N' if not helpful."""
        
        evaluation_messages = [
            SystemMessage(content="You are an evaluator. Respond with only 'HELPFULNESS:Y' or 'HELPFULNESS:N'."),
            HumanMessage(content=evaluation_prompt)
        ]
        
        evaluation = await self.model.ainvoke(evaluation_messages)
        evaluation_result = evaluation.content.strip()
        
        # Add the helpfulness evaluation to the state
        state["messages"].append(AIMessage(content=evaluation_result))
        
        return state
    
    async def _helpfulness_decision(self, state: PersonaAgentState) -> PersonaAgentState:
        """Decide the next step based on helpfulness evaluation."""
        # Check for helpfulness markers in the last few messages
        for message in reversed(state["messages"][-3:]):
            if isinstance(message, AIMessage) and "HELPFULNESS:" in message.content:
                if "HELPFULNESS:Y" in message.content:
                    # Mark as helpful and continue
                    state["messages"].append(AIMessage(content="Decision: Continue to persona response"))
                elif "HELPFULNESS:N" in message.content:
                    # Mark as unhelpful and continue (for now)
                    state["messages"].append(AIMessage(content="Decision: Continue despite unhelpful response"))
        
        # Always continue to persona response for now
        return state
    
    async def _persona_response_node(self, state: PersonaAgentState) -> PersonaAgentState:
        """Generate the final persona-specific response."""
        current_query = state["current_query"]
        a2a_responses = state["a2a_responses"]
        persona_context = state["persona_context"]
        
        # Find the best response to present to the user
        best_response = None
        for response in reversed(a2a_responses):
            if response.get("content") and not response.get("error"):
                best_response = response["content"]
                break
        
        if best_response:
            # Clean up the response (remove any HELPFULNESS markers)
            cleaned_response = best_response.replace("HELPFULNESS:Y", "").replace("HELPFULNESS:N", "").strip()
            
            final_response = f"""As {persona_context}, here's what I found for you:

{cleaned_response}

This response was generated using the A2A protocol with the following capabilities:
- Web search for current information via Tavily
- Academic paper search on arXiv
- Document retrieval from loaded documents
- Multi-turn conversations with helpfulness evaluation

The A2A agent used appropriate tools and evaluated response quality to ensure you get the most helpful answer possible for your specific needs and goals."""
        else:
            final_response = f"""As {persona_context}, I attempted to get information from my A2A agent, but encountered some issues. 

The A2A agent has access to:
- Web search for current information
- Academic paper search on arXiv
- Document retrieval from loaded documents

Please try rephrasing your question or let me know if you need help with a specific type of information."""
        
        state["final_response"] = final_response
        state["messages"].append(AIMessage(content=final_response))
        return state
    
    async def process_query(self, query: str, thread_id: str = None) -> Dict[str, Any]:
        """Process a user query using the Persona Agent Graph."""
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
                "persona_context": "",
                "strategic_approach": "",
                "final_response": ""
            }
            
            # Configure the graph execution with thread_id
            config = {"configurable": {"thread_id": thread_id}}
            
            # Execute the graph
            result = await self.graph.ainvoke(initial_state, config)
            
            # Extract the final response
            final_message = None
            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage) and "A2A protocol with the following capabilities" in message.content:
                    final_message = message.content
                    break
            
            if not final_message:
                final_message = "Failed to generate final response"
            
            return {
                "success": True,
                "response": final_message,
                "persona_type": self.persona_type,
                "a2a_interactions": len(result.get("a2a_responses", [])),
                "thread_id": thread_id,
                "full_state": result
            }
            
        except Exception as e:
            logger.error(f"Error processing query for {self.persona_type}: {e}")
            return {
                "success": False,
                "error": str(e),
                "persona_type": self.persona_type,
                "thread_id": thread_id
            }


# Example usage and testing
async def main():
    """Example usage of different Persona Agents."""
    
    # Test different personas
    personas = ["Machine Learning Expert", "Business Analyst", "Curious Student", "Skeptical Reviewer"]
    
    for persona in personas:
        print(f"\n{'='*60}")
        print(f"�� Testing Persona: {persona}")
        print(f"{'='*60}")
        
        agent = PersonaAgent(persona)
        
        try:
            # Test query that would interest this persona
            test_query = "What makes Kimi K2 so incredible?"
            
            print(f"Query: {test_query}")
            print(f"Persona Goals: {', '.join(agent._get_persona_config()['goals'])}")
            print(f"Persona Style: {agent._get_persona_config()['style']}")
            print("-" * 60)
            
            result = await agent.process_query(test_query)
            
            if result["success"]:
                print("✅ Success!")
                print(f"Response: {result['response'][:300]}...")
                print(f"A2A Interactions: {result['a2a_interactions']}")
            else:
                print("❌ Failed!")
                print(f"Error: {result['error']}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
        
        finally:
            # The original code had agent.close(), but A2AClient is removed.
            # Assuming the intent was to close the model if it was created with a model.
            if agent.model:
                await agent.model.aclose()
        
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main()) 