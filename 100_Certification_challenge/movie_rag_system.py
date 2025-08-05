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

def estimate_tokens(text):
    """Estimate token count for text (rough approximation)"""
    # Rough approximation: 1 token ≈ 4 characters for English text
    return len(text) // 4

def filter_movies_by_review_count(df, min_reviews=5):
    """Filter movies to only include those with at least min_reviews reviews"""
    print(f"🔍 Filtering movies with at least {min_reviews} reviews...")
    
    # Count reviews per movie
    review_counts = df.groupby('id').size().reset_index(name='review_count')
    
    # Filter movies that meet the minimum review count
    qualified_movies = review_counts[review_counts['review_count'] >= min_reviews]
    
    # Filter the original dataframe to only include qualified movies
    filtered_df = df[df['id'].isin(qualified_movies['id'])]
    
    print(f"✅ Filtered dataset: {len(filtered_df)} reviews from {len(qualified_movies)} movies (min {min_reviews} reviews each)")
    print(f"📊 Review count distribution:")
    review_distribution = qualified_movies['review_count'].describe()
    for stat, value in review_distribution.items():
        print(f"   {stat}: {value:.1f}")
    
    return filtered_df, qualified_movies

def create_review_documents(df, max_movies=5000, min_reviews=5):
    """Convert merged DataFrame to list of movie documents (grouped by movie)"""
    documents = []
    
    # First filter by review count
    filtered_df, qualified_movies = filter_movies_by_review_count(df, min_reviews)
    
    # Group by movie ID to collect all reviews for each movie
    print("🎬 Grouping reviews by movie...")
    movie_groups = filtered_df.groupby('id')
    
    # Sort movies by review count (descending) to prioritize movies with more reviews
    movie_review_counts = qualified_movies.sort_values('review_count', ascending=False)
    
    # Limit the number of movies for performance
    movie_count = 0
    processed_movies = 0
    
    for movie_id in movie_review_counts['id']:
        if movie_count >= max_movies:
            break
            
        movie_reviews = movie_groups.get_group(movie_id)
        processed_movies += 1
        
        # Get the first row for movie metadata (all rows have same movie info)
        first_review = movie_reviews.iloc[0]
        
        # Create comprehensive metadata for the movie
        metadata = {
            'source': 'rotten_tomatoes',
            'movie_id': movie_id,
            'movie_title': first_review.get('title_clean', str(movie_id)),
            'genre': first_review.get('genre_clean', ''),
            'director': first_review.get('director_clean', ''),
            'rating': first_review.get('rating', ''),
            'audience_score': first_review.get('audienceScore', ''),
            'tomato_meter': first_review.get('tomatoMeter', ''),
            'release_date': first_review.get('releaseDateTheaters', ''),
            'runtime': first_review.get('runtimeMinutes', ''),
            'total_reviews': len(movie_reviews),
            'review_count': len(movie_reviews)
        }
        
        # Create rich content for embedding - start with movie info
        content = f"Movie: {first_review.get('title_clean', str(movie_id))}\n"
        
        # Add movie metadata
        if first_review.get('genre_clean'):
            content += f"Genre: {first_review.get('genre_clean')}\n"
        if first_review.get('director_clean'):
            content += f"Director: {first_review.get('director_clean')}\n"
        if first_review.get('rating'):
            content += f"Rating: {first_review.get('rating')}\n"
        if first_review.get('releaseDateTheaters'):
            content += f"Release Date: {first_review.get('releaseDateTheaters')}\n"
        if first_review.get('audienceScore'):
            content += f"Audience Score: {first_review.get('audienceScore')}%\n"
        if first_review.get('tomatoMeter'):
            content += f"Tomato Meter: {first_review.get('tomatoMeter')}%\n"
        if first_review.get('runtimeMinutes'):
            content += f"Runtime: {first_review.get('runtimeMinutes')} minutes\n"
        
        # Add review count
        content += f"Total Reviews: {len(movie_reviews)}\n\n"
        
        # Add all reviews for this movie (with character limits)
        content += "Reviews:\n"
        review_count = 0
        total_chars = len(content)  # Start with the content we've already added
        max_chars_per_movie = 10000  # Limit total characters per movie (cut in half)
        print(f"DEBUG: Starting review section for '{first_review.get('title_clean', str(movie_id))}' with {total_chars} chars already")
        
        for idx, review in movie_reviews.iterrows():
            # Check if we're approaching the character limit
            if total_chars > max_chars_per_movie:
                remaining_reviews = len(movie_reviews) - review_count
                content += f"\n... [Additional {remaining_reviews} reviews truncated due to character limit]\n"
                break
                
            # Calculate how many characters this review will add
            critic_name = review.get('criticName_clean', 'Anonymous')
            publication = review.get('publicatioName_clean', 'Unknown')
            review_text = review.get('reviewText_clean', '')
            
            # Estimate characters this review will add
            review_header = f"\n--- Review {idx + 1} ---\n"
            critic_line = f"Critic: {critic_name}"
            if publication != 'Unknown':
                critic_line += f" ({publication})"
            critic_line += "\n"
            
            score_lines = ""
            if review.get('originalScore'):
                score_lines += f"Score: {review.get('originalScore')}\n"
            if review.get('scoreSentiment'):
                score_lines += f"Sentiment: {review.get('scoreSentiment')}\n"
            if review.get('reviewState'):
                score_lines += f"Status: {review.get('reviewState')}\n"
            
            # Truncate review text if needed
            if review_text and len(review_text) > 250:
                review_text = review_text[:250] + "..."
            
            review_content = f"Review: {review_text}\n" if review_text else ""
            
            # Calculate total characters for this review
            review_chars = len(review_header) + len(critic_line) + len(score_lines) + len(review_content)
            
            # Check if adding this review would exceed the limit
            if total_chars + review_chars > max_chars_per_movie:
                remaining_reviews = len(movie_reviews) - review_count
                content += f"\n... [Additional {remaining_reviews} reviews truncated due to character limit]\n"
                break
            
            # Add the review content
            content += review_header
            content += critic_line
            content += score_lines
            content += review_content
            
            review_count += 1
            total_chars += review_chars
            
            # Debug: Check character count periodically
            if review_count % 10 == 0:
                print(f"DEBUG: Movie {first_review.get('title_clean', str(movie_id))} - {review_count} reviews, {total_chars} chars")
        

            
        # Debug: Check final document size
        final_chars = len(content)
        final_tokens = estimate_tokens(content)
        print(f"DEBUG: Final movie document '{first_review.get('title_clean', str(movie_id))}': {final_chars} chars, {final_tokens} tokens")
        
        documents.append({
            'content': content,
            'metadata': metadata
        })
        
        movie_count += 1
    
    print(f"✅ Created {len(documents)} movie documents from {movie_count} movies")
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
        base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return base_retriever

def create_agent_tools():
    """Create the agent's specialized tools"""
    global agent_tools
    
    # Tool 1: Movie Review Search Tool
    def search_movie_reviews(query: str) -> str:
        """
        Search through embedded movie documents from Rotten Tomatoes.
        Each result contains a complete movie with all its reviews, metadata, and ratings.
        Use this for questions about specific movies, ratings, or review content.
        This is the primary tool for finding detailed information about movies in our database.
        """
        try:
            # Use the current base retriever
            retriever = get_base_retriever()
            docs = retriever.invoke(query)
            
            if not docs:
                return f"No relevant movie reviews found for: {query}"
            
            # Format results
            results = f"Found {len(docs)} relevant movie reviews for '{query}':\n\n"
            
            # Track total content length (token safety already handled by TokenSafeRetriever)
            total_tokens = 0
            max_tokens_per_result = 2000  # Conservative limit per result for final formatting
            max_total_tokens = 6000  # Total limit for all results (3 docs × 2k tokens)
            
            # Add info about token management
            results += "ℹ️ Note: Results are automatically token-managed for optimal performance. Documents may be truncated.\n\n"
            
            for i, doc in enumerate(docs, 1):
                metadata = doc.metadata
                content = doc.page_content
                
                results += f"📽️ Result {i}:\n"
                results += f"Movie: {metadata.get('movie_title', 'Unknown')}\n"
                if metadata.get('genre'):
                    results += f"Genre: {metadata.get('genre')}\n"
                if metadata.get('director'):
                    results += f"Director: {metadata.get('director')}\n"
                if metadata.get('audience_score'):
                    results += f"Audience Score: {metadata.get('audience_score')}%\n"
                if metadata.get('tomato_meter'):
                    results += f"Tomato Meter: {metadata.get('tomato_meter')}%\n"
                if metadata.get('total_reviews'):
                    results += f"Total Reviews: {metadata.get('total_reviews')}\n"
                
                # Estimate tokens and truncate content to avoid token limits
                content_tokens = estimate_tokens(content)
                if content_tokens > max_tokens_per_result:
                    # Truncate to approximately max_tokens_per_result tokens
                    max_chars = max_tokens_per_result * 4  # Rough conversion back to chars
                    content = content[:max_chars] + "... [Content truncated due to token limits]"
                    content_tokens = max_tokens_per_result
                
                results += f"Content: {content}\n\n"
                total_tokens += content_tokens
                
                # Stop if we're approaching the total limit
                if total_tokens > max_total_tokens:
                    results += f"... [Additional results truncated due to token limits]\n"
                    break
            
            return results
            
        except Exception as e:
            return f"Error searching reviews: {str(e)}"

    # Tool 2: Movie Statistics Analysis Tool
    def analyze_movie_statistics(movie_name: str = "") -> str:
        """
        Analyze statistics for a specific movie or provide comprehensive Rotten Tomatoes dataset statistics.
        When no movie name is provided, returns detailed dataset analysis including genre distribution,
        director rankings, score analysis, and review statistics.
        Use this for questions about movie trends, dataset overview, or specific movie statistics.
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
                # Comprehensive dataset statistics
                stats = f"🍅 Rotten Tomatoes Dataset Statistics:\n"
                stats += f"═══════════════════════════════════\n"
                
                # Overview
                stats += f"📊 Overview:\n"
                stats += f"• Total Movies in Dataset: {len(movies_df):,}\n"
                stats += f"• Total Reviews in Dataset: {len(reviews_df):,}\n"
                stats += f"• Reviews in Current Sample: {len(merged_df):,}\n"
                stats += f"• Average Reviews per Movie: {len(reviews_df)/len(movies_df):.1f}\n"
                
                # Filtering Information
                movie_review_counts = merged_df.groupby('id').size()
                qualified_movies = movie_review_counts[movie_review_counts >= 5]
                stats += f"\n🔍 Filtering Applied:\n"
                stats += f"• Minimum Reviews Required: 5\n"
                stats += f"• Movies with 5+ reviews: {len(qualified_movies):,}\n"
                stats += f"• Movies excluded (<5 reviews): {len(movie_review_counts) - len(qualified_movies):,}\n"
                stats += f"• Reviews from qualified movies: {qualified_movies.sum():,}\n"
                
                # Review Distribution Analysis
                stats += f"\n📝 Review Distribution:\n"
                stats += f"• Movies with 1-4 reviews: {len(movie_review_counts[movie_review_counts < 5]):,}\n"
                stats += f"• Movies with 5+ reviews: {len(movie_review_counts[movie_review_counts >= 5]):,}\n"
                stats += f"• Movies with 10+ reviews: {len(movie_review_counts[movie_review_counts >= 10]):,}\n"
                stats += f"• Movies with 20+ reviews: {len(movie_review_counts[movie_review_counts >= 20]):,}\n"
                stats += f"• Movies with 50+ reviews: {len(movie_review_counts[movie_review_counts >= 50]):,}\n"
                stats += f"• Movies with 100+ reviews: {len(movie_review_counts[movie_review_counts >= 100]):,}\n"
                
                # Genre Analysis
                if 'genre_clean' in movies_df.columns:
                    stats += f"\n🎭 Genre Distribution (Top 10):\n"
                    genre_counts = movies_df['genre_clean'].value_counts().head(10)
                    for genre, count in genre_counts.items():
                        percentage = (count / len(movies_df)) * 100
                        stats += f"• {genre}: {count:,} movies ({percentage:.1f}%)\n"
                
                # Director Analysis
                if 'director_clean' in movies_df.columns:
                    stats += f"\n🎬 Top Directors (by number of movies):\n"
                    director_counts = movies_df['director_clean'].value_counts().head(10)
                    for director, count in director_counts.items():
                        stats += f"• {director}: {count} movies\n"
                
                # Rating Analysis
                if 'rating' in movies_df.columns:
                    stats += f"\n🏷️ Rating Distribution:\n"
                    rating_counts = movies_df['rating'].value_counts()
                    for rating, count in rating_counts.items():
                        percentage = (count / len(movies_df)) * 100
                        stats += f"• {rating}: {count:,} movies ({percentage:.1f}%)\n"
                
                # Score Analysis
                if 'audienceScore' in movies_df.columns:
                    audience_scores = movies_df['audienceScore'].dropna()
                    if len(audience_scores) > 0:
                        stats += f"\n👥 Audience Score Analysis:\n"
                        stats += f"• Average: {audience_scores.mean():.1f}%\n"
                        stats += f"• Median: {audience_scores.median():.1f}%\n"
                        stats += f"• Highest: {audience_scores.max():.1f}%\n"
                        stats += f"• Lowest: {audience_scores.min():.1f}%\n"
                
                if 'tomatoMeter' in movies_df.columns:
                    tomato_scores = movies_df['tomatoMeter'].dropna()
                    if len(tomato_scores) > 0:
                        stats += f"\n🍅 Tomatometer Analysis:\n"
                        stats += f"• Average: {tomato_scores.mean():.1f}%\n"
                        stats += f"• Median: {tomato_scores.median():.1f}%\n"
                        stats += f"• Highest: {tomato_scores.max():.1f}%\n"
                        stats += f"• Lowest: {tomato_scores.min():.1f}%\n"
                
                # Review Analysis
                if 'reviewState' in reviews_df.columns:
                    stats += f"\n📝 Review State Distribution:\n"
                    review_states = reviews_df['reviewState'].value_counts()
                    for state, count in review_states.items():
                        percentage = (count / len(reviews_df)) * 100
                        stats += f"• {state.title()}: {count:,} reviews ({percentage:.1f}%)\n"
                
                # Top Critics Analysis
                if 'isTopCritic' in reviews_df.columns:
                    top_critic_reviews = reviews_df[reviews_df['isTopCritic'] == True]
                    stats += f"\n👑 Top Critic Analysis:\n"
                    stats += f"• Top Critic Reviews: {len(top_critic_reviews):,}\n"
                    stats += f"• Percentage of Total: {(len(top_critic_reviews)/len(reviews_df)*100):.1f}%\n"
                
                # Top 10 Rated Movies (with minimum review count)
                if 'audienceScore' in movies_df.columns:
                    # Filter movies with at least 10 reviews for more reliable scores
                    movies_with_reviews = merged_df.groupby('id').size().reset_index(name='review_count')
                    movies_with_reviews = movies_with_reviews[movies_with_reviews['review_count'] >= 10]
                    
                    # Get movies that have sufficient reviews
                    reliable_movies = movies_df[movies_df['id'].isin(movies_with_reviews['id'])]
                    reliable_movies = reliable_movies[reliable_movies['audienceScore'].notna()]
                    
                    if len(reliable_movies) >= 10:
                        top_movies = reliable_movies.nlargest(10, 'audienceScore')
                        stats += f"\n🏆 Top 10 Highest Rated Movies (by Audience Score, min 10 reviews):\n"
                        for idx, movie in top_movies.iterrows():
                            title = movie.get('title_clean', 'Unknown')
                            score = movie.get('audienceScore', 0)
                            movie_id = movie.get('id')
                            review_count = movies_with_reviews[movies_with_reviews['id'] == movie_id]['review_count'].iloc[0] if movie_id in movies_with_reviews['id'].values else 0
                            # Safe year extraction
                            release_date = movie.get('releaseDateTheaters', '')
                            if pd.notna(release_date) and isinstance(release_date, str) and len(release_date) >= 4:
                                year = release_date[:4]
                            else:
                                year = 'Unknown'
                            stats += f"• {title} ({year}): {score}% ({review_count} reviews)\n"
                    else:
                        stats += f"\n🏆 Top Rated Movies (by Audience Score):\n"
                        stats += f"• Not enough movies with sufficient reviews for reliable ranking\n"
                
                # Bottom 10 Rated Movies (with minimum review count)
                if 'audienceScore' in movies_df.columns:
                    if len(reliable_movies) >= 10:
                        bottom_movies = reliable_movies.nsmallest(10, 'audienceScore')
                        stats += f"\n💥 Bottom 10 Lowest Rated Movies (by Audience Score, min 10 reviews):\n"
                        for idx, movie in bottom_movies.iterrows():
                            title = movie.get('title_clean', 'Unknown')
                            score = movie.get('audienceScore', 0)
                            movie_id = movie.get('id')
                            review_count = movies_with_reviews[movies_with_reviews['id'] == movie_id]['review_count'].iloc[0] if movie_id in movies_with_reviews['id'].values else 0
                            # Safe year extraction
                            release_date = movie.get('releaseDateTheaters', '')
                            if pd.notna(release_date) and isinstance(release_date, str) and len(release_date) >= 4:
                                year = release_date[:4]
                            else:
                                year = 'Unknown'
                            stats += f"• {title} ({year}): {score}% ({review_count} reviews)\n"
                    else:
                        stats += f"\n💥 Bottom Rated Movies (by Audience Score):\n"
                        stats += f"• Not enough movies with sufficient reviews for reliable ranking\n"
                
                # Top 10 by Tomatometer (with minimum review count)
                if 'tomatoMeter' in movies_df.columns:
                    # Use the same reliable movies filter for Tomatometer
                    reliable_tomato_movies = movies_df[movies_df['id'].isin(movies_with_reviews['id'])]
                    reliable_tomato_movies = reliable_tomato_movies[reliable_tomato_movies['tomatoMeter'].notna()]
                    
                    if len(reliable_tomato_movies) >= 10:
                        top_tomato = reliable_tomato_movies.nlargest(10, 'tomatoMeter')
                        stats += f"\n🍅 Top 10 Highest Rated Movies (by Tomatometer, min 10 reviews):\n"
                        for idx, movie in top_tomato.iterrows():
                            title = movie.get('title_clean', 'Unknown')
                            score = movie.get('tomatoMeter', 0)
                            movie_id = movie.get('id')
                            review_count = movies_with_reviews[movies_with_reviews['id'] == movie_id]['review_count'].iloc[0] if movie_id in movies_with_reviews['id'].values else 0
                            # Safe year extraction
                            release_date = movie.get('releaseDateTheaters', '')
                            if pd.notna(release_date) and isinstance(release_date, str) and len(release_date) >= 4:
                                year = release_date[:4]
                            else:
                                year = 'Unknown'
                            stats += f"• {title} ({year}): {score}% ({review_count} reviews)\n"
                    else:
                        stats += f"\n🍅 Top Rated Movies (by Tomatometer):\n"
                        stats += f"• Not enough movies with sufficient reviews for reliable ranking\n"
                
                # Bottom 10 by Tomatometer (with minimum review count)
                if 'tomatoMeter' in movies_df.columns:
                    if len(reliable_tomato_movies) >= 10:
                        bottom_tomato = reliable_tomato_movies.nsmallest(10, 'tomatoMeter')
                        stats += f"\n🍅 Bottom 10 Lowest Rated Movies (by Tomatometer, min 10 reviews):\n"
                        for idx, movie in bottom_tomato.iterrows():
                            title = movie.get('title_clean', 'Unknown')
                            score = movie.get('tomatoMeter', 0)
                            movie_id = movie.get('id')
                            review_count = movies_with_reviews[movies_with_reviews['id'] == movie_id]['review_count'].iloc[0] if movie_id in movies_with_reviews['id'].values else 0
                            # Safe year extraction
                            release_date = movie.get('releaseDateTheaters', '')
                            if pd.notna(release_date) and isinstance(release_date, str) and len(release_date) >= 4:
                                year = release_date[:4]
                            else:
                                year = 'Unknown'
                            stats += f"• {title} ({year}): {score}% ({review_count} reviews)\n"
                    else:
                        stats += f"\n🍅 Bottom Rated Movies (by Tomatometer):\n"
                        stats += f"• Not enough movies with sufficient reviews for reliable ranking\n"
                
                return stats
                
        except Exception as e:
            return f"Error analyzing statistics: {str(e)}"

    # Tool 3: Smart External Movie Search
    def search_external_movie_info(query: str) -> str:
        """
        Search external sites for movie information when RAG results don't match the query.
        This tool is designed to be used when the retrieved documents don't contain
        information about the specific movie being asked about.
        """
        try:
            # First, try to get RAG results to see if we have relevant info
            retriever = get_base_retriever()
            rag_docs = retriever.invoke(query)
            
            # Check if RAG results are relevant to the query
            query_lower = query.lower()
            relevant_rag_results = False
            
            for doc in rag_docs:
                content_lower = doc.page_content.lower()
                # Check if the query mentions a specific movie and if that movie appears in the RAG results
                if any(word in content_lower for word in query_lower.split() if len(word) > 3):
                    relevant_rag_results = True
                    break
            
            # If RAG results are relevant, suggest using those instead
            if relevant_rag_results:
                return f"RAG results appear to contain relevant information for '{query}'. Consider using the search_movie_reviews tool first to get detailed information from our database."
            
            # If no relevant RAG results, perform external search
            search_string = f'movie {query} reviews ratings news'
            
            if has_tavily:
                result = external_search_tool.invoke({"query": search_string})
                snippets = []
                for item in result[:3]:
                    if isinstance(item, dict):
                        url = item.get("url", "")
                        content = (item.get("content", "") or "").strip()
                        snippets.append(f"Source: {url}\n{content[:300]}…")
                
                if snippets:
                    return f"External search results for '{query}':\n\n" + "\n\n".join(snippets)
                else:
                    return f"No external results found for '{query}'."
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
        max_tokens=4000  # Increased from 1000 to handle larger responses
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

    def truncate_tool_result(result: str, max_tokens: int = 6000) -> str:
        """Truncate tool result to avoid token limits"""
        estimated_tokens = estimate_tokens(result)
        if estimated_tokens > max_tokens:
            # Truncate to approximately max_tokens
            max_chars = max_tokens * 4  # Rough conversion back to chars
            truncated = result[:max_chars] + f"\n\n[Content truncated at {max_tokens} tokens to avoid limits]"
            return truncated
        return result

    def tool_execution_node(state: AgentState) -> AgentState:
        """Execute selected tools with token management"""
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
                        
                        # Debug: Check tool result size
                        result_str = str(result)
                        result_tokens = estimate_tokens(result_str)
                        result_chars = len(result_str)
                        print(f"DEBUG: Tool {tool_name} result: {result_tokens} tokens, {result_chars} chars")
                        
                        # Truncate result to avoid token limits
                        truncated_result = truncate_tool_result(result_str)
                        truncated_tokens = estimate_tokens(truncated_result)
                        print(f"DEBUG: After truncation: {truncated_tokens} tokens")
                        
                        # Create tool message
                        tool_message = ToolMessage(
                            content=truncated_result,
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
    agent_graph.add_node("tools", tool_execution_node)  # Use our custom tool execution
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

def initialize_rag_system(max_movies=5000, min_reviews=5):
    """Initialize the advanced agentic RAG system with configurable filtering"""
    global chat_model, embedding_model, all_documents, chunks, merged_df, movies_df, reviews_df
    global base_retriever, enhanced_agent, query_enhanced_agent_with_tracing
    
    try:
        print("🚀 Initializing Advanced Agentic Movie RAG System...")
        print(f"⚙️ Configuration: max_movies={max_movies}, min_reviews={min_reviews}")
        
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
        
        # Create documents with filtering
        print("📄 Creating movie documents with filtering...")
        all_documents = create_review_documents(merged_df, max_movies=max_movies, min_reviews=min_reviews)
        
        # Convert to LangChain documents
        print("🔪 Using each movie as a separate chunk...")
        chunks = []
        for doc in all_documents:
            langchain_doc = Document(
                page_content=doc['content'],
                metadata=doc['metadata']
            )
            chunks.append(langchain_doc)
        
        print(f"✅ Created {len(chunks)} chunks from {len(all_documents)} movies")
        
        # Initialize models
        print("🧠 Initializing models...")
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=4000)
        
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

def get_filtering_stats():
    """Get statistics about the current filtering applied to the dataset"""
    if merged_df is None:
        return "Dataset not loaded"
    
    movie_review_counts = merged_df.groupby('id').size()
    qualified_movies = movie_review_counts[movie_review_counts >= 5]
    
    return {
        'total_movies_in_dataset': len(movie_review_counts),
        'qualified_movies_5plus_reviews': len(qualified_movies),
        'excluded_movies_less_than_5': len(movie_review_counts) - len(qualified_movies),
        'total_reviews_from_qualified_movies': qualified_movies.sum(),
        'average_reviews_per_qualified_movie': qualified_movies.mean() if len(qualified_movies) > 0 else 0,
        'max_reviews_for_single_movie': qualified_movies.max() if len(qualified_movies) > 0 else 0
    }

def get_token_usage_info():
    """Get information about token usage and limits"""
    return {
        'agent_model_max_tokens': 4000,
        'chat_model_max_tokens': 4000,
        'search_tool_max_tokens_per_result': 2000,
        'search_tool_max_total_tokens': 6000,  # Updated: 3 docs × 2k tokens
        'search_tool_max_results': 3,  # Updated to match actual k value
        'estimated_tokens_per_char': 0.25,  # 1 token ≈ 4 characters
        'max_chars_per_movie': 10000,  # Updated: reduced for better token management
        'max_chars_per_review': 250,
        'max_chars_per_doc_after_retrieval': 8000,  # TokenSafeRetriever truncation limit
        'token_safety_system': 'TokenSafeRetriever',  # New: indicates token safety system in use
        'supports_get_relevant_documents': True,  # New: indicates deprecated method support
        'retriever_methods': ['invoke', 'get_relevant_documents']  # New: supported retrieval methods
    }

def get_document_stats():
    """Get statistics about the current documents"""
    if chunks is None:
        return "Documents not loaded"
    
    token_counts = []
    for i, chunk in enumerate(chunks):
        tokens = estimate_tokens(chunk.page_content)
        token_counts.append(tokens)
    
    return {
        'total_documents': len(chunks),
        'average_tokens_per_document': sum(token_counts) / len(token_counts) if token_counts else 0,
        'max_tokens_in_document': max(token_counts) if token_counts else 0,
        'min_tokens_in_document': min(token_counts) if token_counts else 0,
        'documents_over_10k_tokens': len([t for t in token_counts if t > 10000]),
        'documents_over_15k_tokens': len([t for t in token_counts if t > 15000])
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
    
    if initialize_rag_system(max_movies=5000, min_reviews=5):
        print("✅ Advanced Agentic RAG System is loaded and ready!")
        status = get_system_status()
        for component, loaded in status.items():
            status_icon = "✅" if loaded else "❌"
            print(f"{status_icon} {component}")
        
        # Test query
        print("\n🎬 Testing the system...")
        test_question = "What are some highly rated movies in the database?"
        print(f"Question: {test_question}")
        result = query_movie_agent(test_question)
        if isinstance(result, dict):
            print(f"Answer: {result.get('answer', 'No answer')[:300]}...")
        else:
            print(f"Answer: {str(result)[:300]}...")
    else:
        print("❌ Advanced Agentic RAG System initialization failed") 