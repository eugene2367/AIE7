# Movie Reviews RAG Agentic Solution

An end-to-end RAG (Retrieval Augmented Generation) agentic solution for analyzing and querying movie review datasets from Letterboxd and Metacritic.

## 🎯 Project Overview

This project implements a sophisticated RAG system that combines:
- **Data Processing**: Loading and cleaning movie review data from multiple sources
- **Embedding Generation**: Creating semantic representations of review text
- **Vector Search**: Efficient retrieval of relevant reviews
- **LLM Generation**: Intelligent response generation with context
- **Agentic Features**: Multi-tool reasoning and analysis capabilities

## 📊 Dataset Sources

### Letterboxd Reviews (`letterboxd-reviews.csv`)
- Social media-style reviews with ratings and engagement metrics
- Fields: Movie name, Release Year, Rating, Reviewer name, Review date, Review text, Comment count, Like count

### Metacritic Reviews (`metacritic-reviews.csv`)
- Professional reviews with detailed summaries and dual rating systems
- Fields: Movie name, Release Date, Rating, Summary, User rating, Website rating

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Text Chunking  │    │  Embedding      │
│   (CSV Files)   │───▶│  & Processing   │───▶│  Generation     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │◀───│  RAG Pipeline   │◀───│  Vector Store   │
│   Interface     │    │  (LangGraph)    │    │  (Qdrant)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Technology Stack

### Core Libraries
- **OpenAI**: Embeddings (`text-embedding-3-small`) and LLM generation (`gpt-4o-mini`)
- **LangChain**: RAG framework and document processing
- **LangGraph**: Agent orchestration and workflow management
- **Qdrant**: Vector database for similarity search
- **Pandas**: Data manipulation and preprocessing

### Development Tools
- **UV**: Package management and virtual environment
- **Jupyter**: Interactive development and experimentation
- **Python 3.9+**: Runtime environment

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Navigate to project directory
cd 100_Certification_challenge

# Install dependencies with UV
uv sync

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\\Scripts\\activate   # On Windows
```

### 2. API Key Setup
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the RAG System
```bash
# Start Jupyter notebook
jupyter notebook movie_reviews_rag_system.ipynb
```

### 4. Interactive Usage
```python
# Example usage in Python
from movie_reviews_rag import query_movie_reviews

# Ask questions about movies
result = query_movie_reviews("What do people think about Inception?")
print(result['answer'])
```

## 📋 Features

### ✅ Implemented (Phase 1)
- [x] Data loading and preprocessing from multiple sources
- [x] Text chunking and embedding generation
- [x] Vector database setup with Qdrant
- [x] Basic RAG pipeline with LangGraph
- [x] Interactive query interface
- [x] Multi-source review analysis

### 🚧 Planned (Phase 2)
- [ ] Agentic features with multiple specialized tools
- [ ] Advanced analytics and trend analysis
- [ ] Sentiment analysis and rating correlation
- [ ] Temporal pattern analysis
- [ ] Comparative analysis between sources

### 🔮 Future Enhancements (Phase 3)
- [ ] RAGAS evaluation framework
- [ ] Visualization capabilities
- [ ] Web application deployment
- [ ] Real-time data updates
- [ ] Advanced filtering and search

## 🎯 Query Examples

The system can handle various types of queries:

### Movie-Specific Queries
- "What do people think about Inception?"
- "How was The Dark Knight received?"
- "What are the reviews for Pulp Fiction?"

### Trend Analysis
- "What are the best rated movies?"
- "How do Letterboxd and Metacritic reviews compare?"
- "What are common themes in movie reviews?"

### Comparative Analysis
- "Which movies have the highest ratings on both platforms?"
- "How do social media reviews differ from professional reviews?"

## 📊 Performance Metrics

- **Dataset Size**: ~10,000+ reviews combined
- **Embedding Model**: OpenAI text-embedding-3-small
- **Retrieval Method**: Similarity search (k=5)
- **Response Time**: <5 seconds for typical queries
- **Accuracy**: Context-aware responses with source attribution

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key
LANGSMITH_API_KEY=your_langsmith_key  # Optional
LANGSMITH_TRACING=false  # Set to true for monitoring
```

### Model Configuration
- **Embedding Model**: `text-embedding-3-small`
- **Chat Model**: `gpt-4o-mini`
- **Temperature**: 0.1 (for consistent responses)
- **Max Tokens**: 1000 (configurable)

## 📁 Project Structure

```
100_Certification_challenge/
├── data/
│   ├── letterboxd-reviews.csv
│   └── metacritic-reviews.csv
├── movie_reviews_rag_system.ipynb
├── RAG_AGENTIC_SOLUTION_PLAN.md
├── pyproject.toml
├── README.md
└── .env (create this)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AI Makerspace**: For the comprehensive RAG and agentic development curriculum
- **OpenAI**: For providing the embedding and language models
- **LangChain**: For the excellent RAG framework
- **Qdrant**: For the vector database technology

## 📞 Support

For questions or issues:
1. Check the documentation in `RAG_AGENTIC_SOLUTION_PLAN.md`
2. Review the Jupyter notebook for implementation details
3. Open an issue on the repository

---

**Status**: Phase 1 Complete - Basic RAG system ready for queries! 🎉
