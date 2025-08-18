"""
Persona-Based Agent Graph that demonstrates different agent personas using the A2A protocol.

This module creates specialized agents with different personas and goals that can
interact with the existing A2A protocol implementation, showcasing advanced
agent-to-agent communication patterns.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, TypedDict, Annotated, Literal
from typing_extensions import TypedDict

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.simple_agent_graph import A2AClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PersonaAgentState(TypedDict):
    """State for the Persona Agent Graph."""
    messages: Annotated[List, "messages"]
    a2a_responses: Annotated[List, "a2a_responses"]
    current_query: Annotated[str, "current_query"]
    persona_context: Annotated[str, "persona_context"]
    persona_goals: Annotated[List[str], "persona_goals"]
    persona_style: Annotated[str, "persona_style"]
    interaction_strategy: Annotated[str, "interaction_strategy"]


class PersonaAgent:
    """Persona-based Agent that interacts with the A2A protocol implementation."""
    
    # Predefined personas
    PERSONAS = {
        "ml_expert": {
            "name": "Machine Learning Expert",
            "description": "An expert in machine learning who wants deep, technical information with academic sources",
            "goals": [
                "Get detailed technical explanations",
                "Find academic papers and research",
                "Verify information with credible sources",
                "Understand implementation details"
            ],
            "style": "Technical, thorough, academic-focused",
            "system_prompt": """You are a Machine Learning Expert who is not satisfied with surface-level answers. 
            You want detailed, technical information with academic sources and implementation details.
            
            When interacting with the A2A agent, you should:
            1. Ask for specific technical details
            2. Request academic paper references
            3. Seek implementation examples
            4. Verify claims with credible sources
            5. Ask follow-up questions to deepen understanding
            
            Be persistent in getting comprehensive information."""
        },
        "business_analyst": {
            "name": "Business Analyst",
            "description": "A business analyst who needs practical, actionable insights with market context",
            "goals": [
                "Get practical business insights",
                "Understand market implications",
                "Find actionable recommendations",
                "Get current market data"
            ],
            "style": "Practical, business-focused, results-oriented",
            "system_prompt": """You are a Business Analyst who needs practical, actionable insights.
            
            When interacting with the A2A agent, you should:
            1. Focus on business implications and market impact
            2. Ask for practical applications and use cases
            3. Request current market data and trends
            4. Seek actionable recommendations
            5. Understand ROI and business value
            
            Keep responses focused on business outcomes and practical applications."""
        },
        "curious_student": {
            "name": "Curious Student",
            "description": "A student who wants to learn and understand concepts thoroughly",
            "goals": [
                "Learn fundamental concepts",
                "Get clear explanations",
                "Find learning resources",
                "Understand real-world applications"
            ],
            "style": "Inquisitive, learning-focused, thorough",
            "system_prompt": """You are a Curious Student who wants to learn and understand thoroughly.
            
            When interacting with the A2A agent, you should:
            1. Ask for fundamental explanations of concepts
            2. Request examples and analogies
            3. Seek learning resources and references
            4. Ask about real-world applications
            5. Request clarification when things are unclear
            
            Focus on building a solid foundation of understanding."""
        },
        "skeptical_reviewer": {
            "name": "Skeptical Reviewer",
            "description": "A critical reviewer who questions claims and seeks verification",
            "goals": [
                "Verify claims and statements",
                "Find contradictory evidence",
                "Question assumptions",
                "Get multiple perspectives"
            ],
            "style": "Critical, questioning, verification-focused",
            "system_prompt": """You are a Skeptical Reviewer who questions claims and seeks verification.
            
            When interacting with the A2A agent, you should:
            1. Question claims and ask for evidence
            2. Seek contradictory viewpoints and evidence
            3. Ask about limitations and assumptions
            4. Request multiple sources and perspectives
            5. Challenge conclusions and ask for justification
            
            Be thorough in verifying information and understanding limitations."""
        }
    }
    
    def __init__(self, persona: str, a2a_url: str = "http://localhost:10000"):
        if persona not in self.PERSONAS:
            raise ValueError(f"Unknown persona: {persona}. Available: {list(self.PERSONAS.keys())}")
        
        self.persona_config = self.PERSONAS[persona]
        self.a2a_client = A2AClient(a2a_url)
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2
        )
        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for the Persona Agent."""
        
        # Create the state graph
        workflow = StateGraph(PersonaAgentState)
        
        # Add nodes
        workflow.add_node("persona_analysis", self._persona_analysis_node)
        workflow.add_node("strategic_interaction", self._strategic_interaction_node)
        workflow.add_node("response_evaluation", self._response_evaluation_node)
        workflow.add_node("persona_response", self._persona_response_node)
        
        # Set entry point
        workflow.set_entry_point("persona_analysis")
        
        # Add edges
        workflow.add_edge("persona_analysis", "strategic_interaction")
        workflow.add_edge("strategic_interaction", "response_evaluation")
        workflow.add_edge("response_evaluation", "persona_response")
        workflow.add_edge("persona_response", END)
        
        # Compile the graph
        return workflow.compile(checkpointer=self.memory)
    
    async def _persona_analysis_node(self, state: PersonaAgentState) -> PersonaAgentState:
        """Analyze the query from the persona's perspective."""
        current_query = state["current_query"]
        persona_context = state["persona_context"]
        
        # Create persona-specific analysis
        analysis_prompt = f"""You are {self.persona_config['name']}: {self.persona_config['description']}

Your goals are:
{chr(10).join(f"- {goal}" for goal in self.persona_config['goals'])}

Your style is: {self.persona_config['style']}

Given the user query: "{current_query}"

Analyze this query from your persona's perspective and determine:
1. What specific information you need
2. How you should approach the A2A agent
3. What follow-up questions you should ask
4. What verification or additional details you need

Provide a strategic plan for interacting with the A2A agent."""
        
        analysis_messages = [
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=f"Analyze this query from your persona's perspective: {current_query}")
        ]
        
        response = await self.model.ainvoke(analysis_messages)
        analysis = response.content
        
        # Update state with persona analysis
        state["persona_context"] = analysis
        state["messages"].append(AIMessage(content=f"Persona Analysis: {analysis}"))
        
        return state
    
    async def _strategic_interaction_node(self, state: PersonaAgentState) -> PersonaAgentState:
        """Interact strategically with the A2A agent based on persona goals."""
        current_query = state["current_query"]
        persona_context = state["persona_context"]
        
        # Create a persona-specific query for the A2A agent
        strategic_query = f"""User Query: {current_query}

Persona Context: {self.persona_config['name']} - {self.persona_config['description']}

Persona Goals:
{chr(10).join(f"- {goal}" for goal in self.persona_config['goals'])}

Persona Style: {self.persona_config['style']}

Strategic Approach: {persona_context}

Please provide information that addresses the persona's specific needs and goals. 
Focus on the aspects that matter most to this persona type."""
        
        try:
            # Create a task with the A2A server
            task_result = await self.a2a_client.create_task(strategic_query)
            
            if not task_result:
                # Log the actual error for debugging
                import logging
                logger = logging.getLogger(__name__)
                logger.error("A2A task creation failed - task_result is None")
                
                error_msg = "Failed to create task with A2A server - check server logs for details"
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
                # Log what we actually received
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"A2A response incomplete - task_id: {task_id}, content_length: {len(response_content) if response_content else 0}")
                
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
    
    async def _response_evaluation_node(self, state: PersonaAgentState) -> PersonaAgentState:
        """Fast-pass evaluation: do not trigger follow-ups; proceed to persona response."""
        return state
    
    async def _persona_response_node(self, state: PersonaAgentState) -> PersonaAgentState:
        """Generate the final response in the persona's style."""
        current_query = state["current_query"]
        a2a_responses = state["a2a_responses"]
        persona_context = state["persona_context"]
        
        # Find the best response to present to the user
        best_response = None
        for response in reversed(a2a_responses):
            if response.get("content"):
                best_response = response["content"]
                break
        
        if best_response:
            # Generate a persona-specific final response
            final_response_prompt = f"""You are {self.persona_config['name']} responding to a user.

Your persona: {self.persona_config['description']}
Your style: {self.persona_config['style']}
Your goals: {chr(10).join(f"- {goal}" for goal in self.persona_config['goals'])}

User Query: {current_query}
A2A Agent Response: {best_response}

Generate a final response that:
1. Reflects your persona's style and goals
2. Incorporates the A2A agent's information
3. Addresses the user's query from your perspective
4. Shows how you would typically communicate this information

Keep your response authentic to your persona."""
            
            final_response_messages = [
                SystemMessage(content=f"You are {self.persona_config['name']}."),
                HumanMessage(content=final_response_prompt)
            ]
            
            final_response = await self.model.ainvoke(final_response_messages)
            response_content = final_response.content
        else:
            response_content = f"""As {self.persona_config['name']}, I attempted to get information from my A2A agent, but encountered some issues.

The A2A agent has access to:
- Web search for current information
- Academic paper search on arXiv  
- Document retrieval from loaded documents

From my perspective as {self.persona_config['name']}, I would typically need:
{chr(10).join(f"- {goal}" for goal in self.persona_config['goals'])}

Please try rephrasing your question or let me know if you need help with a specific type of information."""
        
        state["messages"].append(AIMessage(content=response_content))
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
                "persona_goals": self.persona_config["goals"],
                "persona_style": self.persona_config["style"],
                "interaction_strategy": ""
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
                    "persona": self.persona_config["name"],
                    "thread_id": thread_id,
                    "a2a_interactions": len(result["a2a_responses"]),
                    "persona_goals": self.persona_config["goals"]
                }
            else:
                return {
                    "success": False,
                    "error": "No final response generated",
                    "persona": self.persona_config["name"],
                    "thread_id": thread_id
                }
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "persona": self.persona_config["name"],
                "thread_id": thread_id
            }
    
    async def close(self):
        """Clean up resources."""
        await self.a2a_client.close()


# Example usage and testing
async def main():
    """Example usage of different Persona Agents."""
    
    # Test different personas
    personas = ["ml_expert", "business_analyst", "curious_student", "skeptical_reviewer"]
    
    for persona in personas:
        print(f"\n{'='*60}")
        print(f"🧠 Testing Persona: {PersonaAgent.PERSONAS[persona]['name']}")
        print(f"{'='*60}")
        
        agent = PersonaAgent(persona)
        
        try:
            # Test query that would interest this persona
            test_query = "What makes Kimi K2 so incredible?"
            
            print(f"Query: {test_query}")
            print(f"Persona Goals: {', '.join(agent.persona_config['goals'])}")
            print(f"Persona Style: {agent.persona_config['style']}")
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
            await agent.close()
        
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main()) 