"""
Movie RAG System Module - Advanced Agentic Implementation
Standalone implementation that can be used by the Streamlit frontend.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from uuid import uuid4
from dotenv import load_dotenv
import warnings
import asyncio
import nest_asyncio
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup LangSmith environment variables early
langsmith_key = os.getenv("LANGSMITH_API_KEY")
if langsmith_key:
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
    # Use a simpler project name without special characters
    os.environ["LANGCHAIN_PROJECT"] = "movie-reviews-rag"

# Apply nest_asyncio for compatibility
nest_asyncio.apply()
warnings.filterwarnings('ignore')

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required libraries
try:
    import tiktoken
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import Qdrant
    from langchain_community.retrievers import BM25Retriever
    from langchain.retrievers.multi_query import MultiQueryRetriever
    from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_cohere import CohereRerank
    from langchain.storage import InMemoryStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient, models
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langgraph.graph import START, StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain_core.tools import tool
    from langchain_community.tools.tavily_search import TavilySearchResults
    from langchain.tools import Tool
    from typing_extensions import TypedDict, Annotated
    from langgraph.graph.message import add_messages
    from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Please install required packages: uv add langchain-openai langchain-community langchain-cohere langchain-qdrant rank-bm25 tavily-python ragas")

# Global variables
chat_model = None
embedding_model = None
all_documents = None
chunks = None
merged_df = None
movies_df = None
reviews_df = None
base_retriever = None
enhanced_agent = None
query_enhanced_agent_with_tracing = None
agent_tools = []
unique_id = None

def load_csv_robust(file_path):
    """Load CSV file with robust error handling"""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def clean_text(text):
    """Clean text data for better processing"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    return text.strip()

def create_review_documents(df, max_reviews=1000):
    """Convert merged DataFrame to list of review documents"""
    documents = []
    
    # Use a sample for better performance
    if len(df) > max_reviews:
        print(f"🧪 Using sample of {max_reviews} reviews...")
        df_sample = df.head(max_reviews)
    else:
        df_sample = df
    
    for idx, row in df_sample.iterrows():
        # Create comprehensive metadata
        metadata = {
            'source': 'rotten_tomatoes',
            'movie_id': row.get('id', ''),
            'movie_title': row.get('title_clean', row.get('id', 'Unknown')),
            'critic_name': row.get('criticName_clean', 'Anonymous'),
            'publication': row.get('publicatioName_clean', 'Unknown'),
            'review_date': row.get('creationDate', 'Unknown'),
            'original_score': row.get('originalScore', ''),
            'review_state': row.get('reviewState', ''),
            'sentiment': row.get('scoreSentiment', ''),
            'is_top_critic': row.get('isTopCritic', False),
            'genre': row.get('genre_clean', ''),
            'director': row.get('director_clean', ''),
            'rating': row.get('rating', ''),
            'audience_score': row.get('audienceScore', ''),
            'tomato_meter': row.get('tomatoMeter', ''),
            'release_date': row.get('releaseDateTheaters', ''),
            'runtime': row.get('runtimeMinutes', ''),
            'index': idx
        }
        
        # Create rich content for embedding
        content = f"Movie: {row.get('title_clean', row.get('id', 'Unknown'))}\n"
        
        # Add movie metadata
        if row.get('genre_clean'):
            content += f"Genre: {row.get('genre_clean')}\n"
        if row.get('director_clean'):
            content += f"Director: {row.get('director_clean')}\n"
        if row.get('rating'):
            content += f"Rating: {row.get('rating')}\n"
        if row.get('releaseDateTheaters'):
            content += f"Release Date: {row.get('releaseDateTheaters')}\n"
        
        # Add review information
        content += f"Critic: {row.get('criticName_clean', 'Anonymous')}\n"
        if row.get('publicatioName_clean'):
            content += f"Publication: {row.get('publicatioName_clean')}\n"
        if row.get('originalScore'):
            content += f"Score: {row.get('originalScore')}\n"
        if row.get('reviewState'):
            content += f"Review State: {row.get('reviewState')}\n"
        if row.get('scoreSentiment'):
            content += f"Sentiment: {row.get('scoreSentiment')}\n"
        
        # Add the main review text
        review_text = row.get('reviewText_clean', '')
        if review_text:
            content += f"Review: {review_text}"
        
        documents.append({
            'content': content,
            'metadata': metadata
        })
    
    return documents

def setup_external_search():
    """Setup external search tools"""
    global external_search_tool, has_tavily
    
    print("🔧 Setting up external search tools...")
    
    # Option 1: Tavily Search (recommended)
    try:
        tavily_search = TavilySearchResults(
            max_results=3,
            search_depth="basic",
            include_answer=True,
            include_raw_content=True
        )
        print("✅ Tavily search tool configured")
        has_tavily = True
        external_search_tool = tavily_search
    except Exception as e:
        print(f"⚠️ Tavily not configured: {e}")
        has_tavily = False
        
        # Create a fallback search function
        def fallback_search(query: str) -> str:
            """Fallback search when no external APIs are available"""
            return f"External search not available. Query '{query}' would require external movie database access. Please configure Tavily API key for enhanced search capabilities."
        
        external_search_tool = Tool(
            name="fallback_search",
            description="Fallback search tool when external APIs are not configured",
            func=fallback_search
        )
    
    search_tool_name = "Tavily" if has_tavily else "Fallback"
    print(f"🔍 Using {search_tool_name} for external search")

def get_base_retriever():
    """Get the base retriever (can be dynamically switched)"""
    global base_retriever
    if base_retriever is None:
        vectorstore = Qdrant.from_documents(
            chunks,
            embedding_model,
            location=":memory:",
            collection_name="MovieReviews_Default"
        )
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    return base_retriever

def create_agent_tools():
    """Create the agent's specialized tools"""
    global agent_tools
    
    # Tool 1: Movie Review Search Tool
    def search_movie_reviews(query: str) -> str:
        """
        Search through embedded movie reviews from Rotten Tomatoes.
        Use this for questions about specific movies, ratings, or review content.
        """
        try:
            # Use the current base retriever
            retriever = get_base_retriever()
            docs = retriever.invoke(query)
            
            if not docs:
                return f"No relevant movie reviews found for: {query}"
            
            # Format results
            results = f"Found {len(docs)} relevant movie reviews for '{query}':\n\n"
            
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                content = doc.page_content
                
                results += f"📽️ Result {i}:\n"
                results += f"Movie: {metadata.get('movie_title', 'Unknown')}\n"
                results += f"Critic: {metadata.get('critic_name', 'Unknown')}\n"
                if metadata.get('publication'):
                    results += f"Publication: {metadata.get('publication')}\n"
                if metadata.get('original_score'):
                    results += f"Score: {metadata.get('original_score')}\n"
                results += f"Content: {content[:200]}...\n\n"
            
            return results
            
        except Exception as e:
            return f"Error searching reviews: {str(e)}"

    # Tool 2: Movie Statistics Analysis Tool
    def analyze_movie_statistics(movie_name: str = "") -> str:
        """
        Analyze statistics for a specific movie or provide general Rotten Tomatoes dataset statistics.
        Returns ratings, review counts, critic information, and other numerical insights.
        """
        try:
            if movie_name:
                # Search for specific movie in the merged dataset
                movie_data = merged_df[
                    merged_df['title_clean'].str.contains(movie_name, case=False, na=False)
                ]
                
                if movie_data.empty:
                    return f"No statistics found for '{movie_name}' in the Rotten Tomatoes dataset."
                
                # Get movie information
                movie_info = movie_data.iloc[0]  # Get first match for movie metadata
                movie_reviews = movie_data  # All reviews for this movie
                
                stats = f"Statistics for '{movie_info.get('title_clean', movie_name)}':\n"
                stats += f"═══════════════════════════════════\n"
                
                # Movie metadata
                if movie_info.get('genre_clean'):
                    stats += f"🎭 Genre: {movie_info['genre_clean']}\n"
                if movie_info.get('director_clean'):
                    stats += f"🎬 Director: {movie_info['director_clean']}\n"
                if movie_info.get('rating'):
                    stats += f"🏷️ Rating: {movie_info['rating']}\n"
                if movie_info.get('runtimeMinutes'):
                    stats += f"⏱️ Runtime: {movie_info['runtimeMinutes']} minutes\n"
                if movie_info.get('releaseDateTheaters'):
                    stats += f"📅 Release Date: {movie_info['releaseDateTheaters']}\n"
                
                # Scores
                if pd.notna(movie_info.get('audienceScore')):
                    stats += f"👥 Audience Score: {movie_info['audienceScore']}%\n"
                if pd.notna(movie_info.get('tomatoMeter')):
                    stats += f"🍅 Tomatometer: {movie_info['tomatoMeter']}%\n"
                
                # Review statistics
                stats += f"\n📊 Review Analysis:\n"
                stats += f"• Total Reviews: {len(movie_reviews)}\n"
                
                # Review state distribution
                if 'reviewState' in movie_reviews.columns:
                    review_states = movie_reviews['reviewState'].value_counts()
                    for state, count in review_states.items():
                        stats += f"• {state.title()}: {count} reviews\n"
                
                return stats
            else:
                # General dataset statistics
                stats = f"🍅 Rotten Tomatoes Dataset Statistics:\n"
                stats += f"═══════════════════════════════════\n"
                stats += f"📊 Overview:\n"
                stats += f"• Total Movies: {len(movies_df):,}\n"
                stats += f"• Total Reviews: {len(reviews_df):,}\n"
                stats += f"• Reviews in Current Sample: {len(merged_df):,}\n"
                stats += f"• Average Reviews per Movie: {len(reviews_df)/len(movies_df):.1f}\n"
                
                return stats
                
        except Exception as e:
            return f"Error analyzing statistics: {str(e)}"

    # Tool 3: External Movie Search
    def search_external_movie_info(query: str) -> str:
        """
        Search external sites for reviews, ratings, or recent news about a movie.
        """
        try:
            search_string = f'movie {query} reviews ratings'

            if has_tavily:
                result = external_search_tool.invoke({"query": search_string})
                snippets = []
                for item in result[:3]:
                    if isinstance(item, dict):
                        url = item.get("url", "")
                        content = (item.get("content", "") or "").strip()
                        snippets.append(f"Source: {url}\n{content[:200]}…")
                return "\n\n".join(snippets) if snippets else "No results found."
            else:
                return external_search_tool.run(search_string)

        except Exception as e:
            return f"External search error: {e}"

    # Create the agent's toolbox with @tool decorators
    agent_tools = [
        tool(search_movie_reviews),
        tool(analyze_movie_statistics), 
        tool(search_external_movie_info)
    ]
    
    print(f"✅ Created {len(agent_tools)} specialized tools:")
    for tool_obj in agent_tools:
        print(f"  - {tool_obj.name}")
    
    print("\n✅ All agent tools ready!")

def create_enhanced_agent():
    """Create the enhanced agent with tool selection"""
    global enhanced_agent, unique_id
    
    print("🤖 Building enhanced agent with tool selection...")
    
    # Enhanced Agent State with Tool Selection
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        question: str
        tool_calls: list
        final_answer: str
    
    # Create the agent prompt
    AGENT_PROMPT = """You are an intelligent movie analysis agent with access to multiple specialized tools.

Your tools:
1. search_movie_reviews: Search embedded movie reviews from Rotten Tomatoes
2. analyze_movie_statistics: Get numerical statistics about movies and datasets  
3. search_external_movie_info: Search external sources when local data is insufficient

Guidelines:
- Start with local review data (search_movie_reviews) for most questions
- Use statistics tools for numerical analysis
- Only use external search when local data is clearly insufficient
- Always explain your reasoning and cite sources
- Provide comprehensive, insightful answers

Current question: {question}
"""

    # Create enhanced chat model with tool binding
    agent_model = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.1,
        max_tokens=1000
    ).bind_tools(agent_tools)

    def agent_reasoning_node(state: AgentState) -> AgentState:
        """Agent reasoning and tool selection"""
        question = state["question"]
        messages = state.get("messages", [])
        
        # Create the prompt with current question
        prompt_message = HumanMessage(content=AGENT_PROMPT.format(question=question))
        
        # Get agent response with potential tool calls
        response = agent_model.invoke([prompt_message] + messages)
        
        return {
            "messages": [response],
            "tool_calls": response.tool_calls if hasattr(response, 'tool_calls') and response.tool_calls else []
        }

    def tool_execution_node(state: AgentState) -> AgentState:
        """Execute selected tools"""
        tool_calls = state.get("tool_calls", [])
        messages = []
        
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            # Find and execute the tool
            for tool in agent_tools:
                if tool.name == tool_name:
                    try:
                        result = tool.invoke(tool_args)
                        # Create tool message
                        tool_message = ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(tool_message)
                    except Exception as e:
                        error_message = ToolMessage(
                            content=f"Error executing {tool_name}: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(error_message)
                    break
        
        return {"messages": messages}

    def final_response_node(state: AgentState) -> AgentState:
        """Generate final response based on tool results"""
        messages = state["messages"]
        question = state["question"]
        
        # Create final prompt
        final_prompt = f"""
        Based on the tool results above, provide a comprehensive answer to the question: {question}
        
        Make sure to:
        - Synthesize information from multiple sources
        - Cite specific data points and sources
        - Provide insights beyond just raw data
        - Be conversational but informative
        """
        
        final_response = chat_model.invoke(messages + [HumanMessage(content=final_prompt)])
        
        return {
            "final_answer": final_response.content,
            "messages": [final_response]
        }

    # Build the enhanced agent graph
    print("🔗 Building agent workflow...")
    
    # Create agent graph
    agent_graph = StateGraph(AgentState)
    
    # Add nodes
    agent_graph.add_node("agent", agent_reasoning_node)
    agent_graph.add_node("tools", ToolNode(agent_tools))
    agent_graph.add_node("final_response", final_response_node)
    
    # Add edges
    agent_graph.add_edge(START, "agent")
    
    # Conditional edge: if agent makes tool calls, go to tools; otherwise go to final response
    def should_continue(state: AgentState) -> str:
        tool_calls = state.get("tool_calls", [])
        if tool_calls:
            return "tools"
        else:
            return "final_response"
    
    agent_graph.add_conditional_edges("agent", should_continue)
    agent_graph.add_edge("tools", "final_response")
    agent_graph.add_edge("final_response", END)
    
    # Compile the enhanced agent
    enhanced_agent = agent_graph.compile()
    
    print("✅ Enhanced agent with tool selection ready!")
    
    # Generate unique project ID for this session
    unique_id = uuid4().hex[:8]
    project_name = f"Movie-Reviews-RAG-{unique_id}"
    
    # Configure LangSmith if available
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_PROJECT"] = project_name
        print(f"🎯 LangSmith project: {project_name}")
    
    print("🚀 Enhanced agent ready for movie analysis!")

def initialize_rag_system():
    """Initialize the advanced agentic RAG system"""
    global chat_model, embedding_model, all_documents, chunks, merged_df, movies_df, reviews_df
    global base_retriever, enhanced_agent, query_enhanced_agent_with_tracing
    
    try:
        print("🚀 Initializing Advanced Agentic Movie RAG System...")
        
        # Load data
        print("📊 Loading movie data...")
        movies_df = load_csv_robust("data/rotten_tomatoes_movies.csv")
        reviews_df = load_csv_robust("data/rotten_tomatoes_movie_reviews.csv")
        
        if movies_df is None or reviews_df is None:
            print("❌ Failed to load data files")
            return False
        
        # Clean and prepare data
        print("🧹 Cleaning and preparing data...")
        movies_df['title_clean'] = movies_df['title'].apply(clean_text)
        movies_df['genre_clean'] = movies_df['genre'].apply(clean_text) 
        movies_df['director_clean'] = movies_df['director'].apply(clean_text)
        
        reviews_df['reviewText_clean'] = reviews_df['reviewText'].apply(clean_text)
        reviews_df['criticName_clean'] = reviews_df['criticName'].apply(clean_text)
        reviews_df['publicatioName_clean'] = reviews_df['publicatioName'].apply(clean_text)
        
        # Merge data
        print("🔗 Merging movies and reviews data...")
        merged_df = reviews_df.merge(
            movies_df, 
            left_on='id', 
            right_on='id', 
            how='left'
        )
        
        print(f"✅ Merged dataset: {len(merged_df)} reviews with movie metadata")
        
        # Create documents
        print("📄 Creating review documents...")
        all_documents = create_review_documents(merged_df, max_reviews=1000)
        
        # Convert to LangChain documents
        print("🔪 Using each review as a separate chunk...")
        chunks = []
        for doc in all_documents:
            langchain_doc = Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            )
            chunks.append(langchain_doc)
        
        print(f"✅ Created {len(chunks)} chunks from {len(all_documents)} reviews")
        
        # Initialize models
        print("🧠 Initializing models...")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Setup LangSmith tracing
        print("🔍 Setting up LangSmith tracing...")
        langsmith_key = os.getenv("LANGSMITH_API_KEY")
        if langsmith_key:
            # Generate a proper UUID for the session
            global unique_id
            unique_id = str(uuid4())
            print("✅ LangSmith tracing configured")
        else:
            print("⚠️ LangSmith API key not found - tracing disabled")
        
        # Setup external search
        setup_external_search()
        
        # Create agent tools
        create_agent_tools()
        
        # Create enhanced agent
        create_enhanced_agent()
        
        # Create the query function
        def query_enhanced_agent_with_tracing(question: str, run_name: str = None) -> dict:
            """Query the enhanced agent with LangSmith tracing"""
            
            # Generate run name if not provided
            if not run_name:
                run_name = f"movie_query_{int(time.time())}"
            
            # Add tags for better organization
            tags = ["movie-reviews", "rag-agent", "multi-tool"]
            
            try:
                # Execute with tracing metadata
                start_time = time.time()
                
                result = enhanced_agent.invoke(
                    {
                        "question": question,
                        "messages": [],
                        "tool_calls": [],
                        "final_answer": ""
                    },
                    config={
                        "tags": tags,
                        "metadata": {
                            "query_type": "movie_analysis",
                            "session_id": unique_id,
                            "run_name": run_name
                        }
                    }
                )
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Return answer and trace info
                return {
                    "answer": result.get("final_answer", "No answer generated"),
                    "run_name": run_name,
                    "execution_time": execution_time
                }
                
            except Exception as e:
                return {
                    "answer": f"Error: {str(e)}",
                    "run_name": run_name,
                    "execution_time": 0
                }
        
        # Make the function globally available
        globals()['query_enhanced_agent_with_tracing'] = query_enhanced_agent_with_tracing
        
        print("✅ Advanced Agentic RAG System initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return False

def is_system_loaded():
    """Check if the RAG system is properly loaded"""
    return all([
        chat_model is not None,
        embedding_model is not None,
        chunks is not None,
        enhanced_agent is not None,
        query_enhanced_agent_with_tracing is not None
    ])

def get_system_status():
    """Get status of all RAG system components"""
    return {
        'chat_model': chat_model is not None,
        'embedding_model': embedding_model is not None,
        'all_documents': all_documents is not None,
        'chunks': len(chunks) if chunks is not None else 0,
        'enhanced_agent': enhanced_agent is not None,
        'query_function': query_enhanced_agent_with_tracing is not None
    }

def query_movie_agent(question: str, retriever_name: str = "naive") -> str:
    """Query function for the frontend - uses the enhanced agentic system"""
    if not is_system_loaded():
        return "❌ RAG System not loaded. Please initialize the system first."
    
    try:
        # Use the enhanced agentic system
        return query_enhanced_agent_with_tracing(question)
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Try to initialize the system automatically
if __name__ == "__main__":
    # Test the module
    print("Movie RAG System Module - Advanced Agentic Implementation")
    print("=" * 60)
    
    if initialize_rag_system():
        print("✅ Advanced Agentic RAG System is loaded and ready!")
        status = get_system_status()
        for component, loaded in status.items():
            status_icon = "✅" if loaded else "❌"
            print(f"{status_icon} {component}")
        
        # Test query
        print("\n🎬 Testing the system...")
        test_question = "What are some highly rated movies in the database?"
        print(f"Question: {test_question}")
        answer = query_movie_agent(test_question)
        print(f"Answer: {answer[:200]}...")
    else:
        print("❌ Advanced Agentic RAG System initialization failed") 