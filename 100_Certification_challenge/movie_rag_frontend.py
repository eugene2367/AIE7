import streamlit as st
import os
import sys
import time
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
import uuid
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from langchain_core.runnables import RunnableConfig

# Add the current directory to path to import from notebook functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import the existing RAG system components
try:
    # Import from the movie_rag_system module
    from movie_rag_system import (
        query_movie_agent,
        initialize_rag_system,
        is_system_loaded,
        get_system_status
    )
    
    # Check for API keys first
    import os
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_key or openai_key == "your_openai_api_key_here":
        st.error("""
        ## 🔑 API Key Required
        
        Please set up your API keys first:
        
        1. **Run the setup script:**
           ```bash
           python setup_env.py
           ```
        
        2. **Or create a .env file manually:**
           ```bash
           echo "OPENAI_API_KEY=your_actual_key_here" > .env
           ```
        
        3. **Then restart this app**
        """)
        st.stop()
    
    # Try to initialize the system if not loaded
    if not is_system_loaded():
        with st.spinner("Initializing RAG system..."):
            initialize_rag_system()
    
    RAG_SYSTEM_LOADED = is_system_loaded()
except ImportError as e:
    RAG_SYSTEM_LOADED = False
    st.error(f"⚠️ RAG system not loaded: {e}")

# Page configuration
st.set_page_config(
    page_title="🎬 Movie Review AI Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for movie theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .retriever-selector {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #FF5252, #26A69A);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_retriever' not in st.session_state:
    st.session_state.current_retriever = "Agentic System"
if 'conversation_stats' not in st.session_state:
    st.session_state.conversation_stats = {
        'total_queries': 0,
        'total_tokens': 0,
        'avg_response_time': 0
    }
if 'example_query' not in st.session_state:
    st.session_state.example_query = None
if 'tracing_enabled' not in st.session_state:
    st.session_state.tracing_enabled = True
if 'trace_links' not in st.session_state:
    st.session_state.trace_links = []

def setup_langsmith_tracing():
    """Setup LangSmith tracing if API key is available"""
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if langsmith_key:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        return True
    return False

def create_trace_config(user_input: str) -> tuple:
    """Create tracing configuration for a query"""
    run_id = str(uuid.uuid4())
    project_name = f"Movie-RAG-Frontend-{datetime.now().strftime('%Y%m%d')}"
    
    config = RunnableConfig(
        tags=["frontend", "movie-rag", "agentic"],
        metadata={
            "user_input": user_input,
            "session_id": st.session_state.get('session_id', 'unknown'),
            "timestamp": datetime.now().isoformat()
        },
        run_name=f"frontend_query_{run_id[:8]}"
    )
    
    # Set project name for this run
    if os.getenv("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_PROJECT"] = project_name
    
    return config, run_id

def get_langsmith_link(run_id: str) -> str:
    """Generate LangSmith link for a run"""
    if os.getenv("LANGSMITH_API_KEY"):
        return f"https://smith.langchain.com/runs/{run_id}"
    return None

def get_langsmith_project_link() -> str:
    """Generate LangSmith project link"""
    if os.getenv("LANGSMITH_API_KEY"):
        # Use your specific workspace ID
        workspace_id = "a8b64252-5f0f-4f35-a048-c004586e098a"
        # LangSmith creates projects with UUIDs, so we'll use a generic link
        # The actual project UUID will be visible in the traces
        return f"https://smith.langchain.com/o/{workspace_id}/projects"
    return None

def get_langsmith_trace_link(project_id: str, trace_id: str) -> str:
    """Generate specific LangSmith trace link"""
    if os.getenv("LANGSMITH_API_KEY"):
        workspace_id = "a8b64252-5f0f-4f35-a048-c004586e098a"
        return f"https://smith.langchain.com/o/{workspace_id}/projects/p/{project_id}?peek={trace_id}&peeked_trace={trace_id}"
    return None

def query_movie_agent_wrapper(user_input: str, retriever_name: str) -> Dict[str, Any]:
    """Query the movie agent with the agentic system and tracing"""
    try:
        start_time = time.time()
        
        # Setup tracing if enabled
        tracing_enabled = st.session_state.get('tracing_enabled', True)
        langsmith_link = None
        run_name = None
        
        if tracing_enabled and os.getenv("LANGSMITH_API_KEY"):
            setup_langsmith_tracing()
            # Generate a unique run name for this query
            timestamp = int(time.time())
            session_short = st.session_state.session_id[:8]  # Use first 8 chars of UUID
            run_name = f"frontend_query_{timestamp}_{session_short}"
            
            try:
                # Import the enhanced agent function for tracing
                from movie_rag_system import query_enhanced_agent_with_tracing
                result = query_enhanced_agent_with_tracing(user_input, run_name)
                
                # Handle new return format
                if isinstance(result, dict):
                    response = result.get("answer", "No answer generated")
                    run_name = result.get("run_name", run_name)
                else:
                    response = result
                
                # Generate LangSmith project link
                langsmith_link = get_langsmith_project_link()
            except Exception as e:
                st.error(f"Tracing error: {str(e)}")
                # Fallback to non-tracing version
                response = query_movie_agent(user_input)
                langsmith_link = None
        else:
            response = query_movie_agent(user_input)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Store trace link if available
        if langsmith_link and tracing_enabled:
            st.session_state.trace_links.append({
                'query': user_input,
                'link': langsmith_link,
                'timestamp': datetime.now().isoformat(),
                'run_name': run_name,
                'response_preview': response[:100] + "..." if len(response) > 100 else response
            })
        
        return {
            'response': response,
            'response_time': response_time,
            'retriever_used': "Agentic System",
            'success': True,
            'trace_id': run_name,
            'langsmith_link': langsmith_link
        }
        
    except Exception as e:
        return {
            'response': f"❌ Error: {str(e)}",
            'response_time': 0,
            'retriever_used': "Agentic System",
            'success': False,
            'trace_id': None,
            'langsmith_link': None
        }



# Main app
def main():
    # Initialize session ID for tracing
    if 'session_id' not in st.session_state:
        # Use a proper UUID for session tracking
        st.session_state.session_id = str(uuid.uuid4())
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 Movie Review AI Assistant</h1>
        <p>Your intelligent companion for exploring movie reviews, ratings, and cinematic insights!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Configuration")
        
        # System info
        st.markdown("#### 🤖 Agentic RAG System")
        st.info("This system uses an advanced agentic RAG approach with multiple specialized tools for comprehensive movie analysis.")
        
        # LangSmith info
        langsmith_key = os.getenv("LANGSMITH_API_KEY")
        
        if langsmith_key:
            st.success("✅ LangSmith tracing active")
            st.info("🔍 Trace links will take you to your LangSmith workspace where you can find detailed agent reasoning, tool calls, and response generation.")
            st.info("💡 Look for the most recent project with movie-related traces.")
            
            # Debug info (expandable)
            with st.expander("🔧 Debug Info"):
                st.write(f"**API Key:** {langsmith_key[:10]}...")
                st.write(f"**Session ID:** {st.session_state.get('session_id', 'N/A')}")
                st.write("**Workspace:** a8b64252-5f0f-4f35-a048-c004586e098a")
                st.write("**Note:** Projects are created dynamically with UUIDs")
            
            # Test tracing button
            if st.button("🧪 Test Tracing"):
                test_result = query_movie_agent_wrapper("What are the best movies?", "test")
                st.success("✅ Tracing test completed! Check the trace links below.")
        else:
            st.warning("⚠️ LangSmith API key not found")
            st.info("Set LANGSMITH_API_KEY environment variable to enable detailed tracing.")
        
        # System status
        st.markdown("#### 📊 System Status")
        if RAG_SYSTEM_LOADED:
            st.success("✅ RAG System Loaded")
            status = get_system_status()
            st.info(f"📚 {status.get('chunks', 0)} document chunks available")
        else:
            st.error("❌ RAG System Not Loaded")
            st.info("The system will attempt to initialize automatically")
        
        # Conversation stats
        st.markdown("#### 📈 Conversation Stats")
        stats = st.session_state.conversation_stats
        st.metric("Total Queries", stats['total_queries'])
        st.metric("Avg Response Time", f"{stats['avg_response_time']:.2f}s")
        
        # Tracing controls
        st.markdown("#### 🔍 Tracing & Debugging")
        tracing_enabled = st.checkbox(
            "Enable LangSmith Tracing", 
            value=st.session_state.get('tracing_enabled', True),
            help="Enable detailed tracing to see how the agent works"
        )
        st.session_state.tracing_enabled = tracing_enabled
        
        # Show trace links if available
        if st.session_state.trace_links:
            st.markdown("#### 📊 Recent Traces")
            for trace in st.session_state.trace_links[-3:]:  # Show last 3 traces
                with st.expander(f"🔍 {trace['query'][:50]}..."):
                    st.write(f"**Query:** {trace['query']}")
                    st.write(f"**Response:** {trace.get('response_preview', 'N/A')}")
                    st.write(f"**Time:** {trace['timestamp']}")
                    if trace.get('run_name'):
                        st.write(f"**Run Name:** {trace['run_name']}")
                    st.markdown(f"[🔗 View Project in LangSmith]({trace['link']})")
                    st.info("💡 Look for the run with the matching run name in the project page.")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_stats = {
                'total_queries': 0,
                'total_tokens': 0,
                'avg_response_time': 0
            }
            st.session_state.trace_links = []
            st.rerun()
    
    # Main chat area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 💬 Chat with Movie AI")
        
        # Display chat messages using Streamlit's native chat components
        for message in st.session_state.messages:
            if message['role'] == "user":
                with st.chat_message("user", avatar="🎭"):
                    st.write(message['content'])
            else:
                with st.chat_message("assistant", avatar="🎬"):
                    # Add metadata if available
                    if message.get('metadata', {}).get('retriever_used'):
                        st.caption(f"🔍 Using {message['metadata']['retriever_used']} retriever • ⏱️ {message['metadata'].get('response_time', 0):.2f}s")
                    
                    # Show tracing info if available
                    if message.get('metadata', {}).get('langsmith_link'):
                        st.caption(f"🔗 [View in LangSmith]({message['metadata']['langsmith_link']})")
                    
                    st.write(message['content'])
        
        # Chat input
        user_input = st.chat_input("Ask about movies, reviews, ratings, or cinematic insights...")
        
        # Handle example query if set
        if st.session_state.example_query:
            user_input = st.session_state.example_query
            st.session_state.example_query = None  # Clear after use
        
        if user_input:
            # Add user message
            st.session_state.messages.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now()
            })
            
            # Get AI response
            with st.spinner("🎬 Movie AI is thinking..."):
                result = query_movie_agent_wrapper(user_input, st.session_state.current_retriever)
                
                # Add assistant message
                st.session_state.messages.append({
                    'role': 'assistant',
                    'content': result['response'],
                    'metadata': {
                        'retriever_used': result['retriever_used'],
                        'response_time': result['response_time'],
                        'trace_id': result.get('trace_id'),
                        'langsmith_link': result.get('langsmith_link')
                    },
                    'timestamp': datetime.now()
                })
                
                # Update stats
                st.session_state.conversation_stats['total_queries'] += 1
                if st.session_state.conversation_stats['total_queries'] > 1:
                    current_avg = st.session_state.conversation_stats['avg_response_time']
                    new_avg = (current_avg + result['response_time']) / 2
                    st.session_state.conversation_stats['avg_response_time'] = new_avg
                else:
                    st.session_state.conversation_stats['avg_response_time'] = result['response_time']
                
                # Rerun to update the display
                st.rerun()
    
    with col2:
        st.markdown("### 🎭 Quick Actions")
        
        # Example queries
        st.markdown("#### 💡 Try These Questions:")
        example_queries = [
            "What are the best rated movies in the database?",
            "Tell me about the reviews for 'The Shawshank Redemption'",
            "What movies have the highest audience scores?",
            "Show me movies directed by Christopher Nolan",
            "What are the most popular genres?",
            "Tell me about recent blockbuster movies"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}"):
                st.session_state.example_query = query
                st.rerun()
        
        # System info
        st.markdown("#### ℹ️ About This AI")
        st.info("""
        This AI assistant uses advanced RAG (Retrieval Augmented Generation) 
        to provide insights about movies, reviews, and ratings from the 
        Rotten Tomatoes database.
        
        **Features:**
        - 🎬 Movie reviews and ratings
        - 📊 Critic and audience scores
        - 🎭 Genre and director analysis
        - 🔍 Advanced search capabilities
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🎬 Movie Review AI Assistant | Powered by LangChain & OpenAI
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    if not RAG_SYSTEM_LOADED:
        st.error("""
        ## ⚠️ Setup Required
        
        The RAG system is attempting to initialize automatically. If you see this message, it means:
        
        1. **The system is still loading** - please wait a moment
        2. **API keys may be missing** - check your environment variables
        3. **Data files may be missing** - ensure the data directory exists
        
        **Troubleshooting:**
        - Check that your OpenAI API key is set: `export OPENAI_API_KEY=your_key`
        - Ensure the data files exist: `data/rotten_tomatoes_movies.csv` and `data/rotten_tomatoes_movie_reviews.csv`
        - Try refreshing the page if the system doesn't load automatically
        """)
    else:
        main() 