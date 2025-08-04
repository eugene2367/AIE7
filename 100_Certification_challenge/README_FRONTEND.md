# 🎬 Movie Review AI Assistant - Frontend

A beautiful, movie-themed Streamlit frontend for interacting with your agentic RAG system. This frontend provides an intuitive chat interface for exploring movie reviews, ratings, and cinematic insights.

## ✨ Features

- **🎭 Movie-Themed UI**: Beautiful gradient design with movie icons and colors
- **💬 Interactive Chat**: Real-time conversation with the AI assistant
- **🔍 Multiple Retrievers**: Switch between 6 different retriever methods
- **📊 Live Statistics**: Track queries, response times, and system status
- **🎯 Quick Actions**: Pre-built example queries for easy testing
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Quick Start

### Prerequisites

1. **Run the RAG System Notebook First**
   ```bash
   # Make sure you're in the 100_Certification_challenge directory
   cd 100_Certification_challenge
   
   # Start Jupyter notebook
   jupyter notebook
   ```

2. **Execute the RAG System**
   - Open `Movie_RAG_System.ipynb`
   - Run ALL cells (Cell → Run All)
   - **Keep the notebook running** (don't close it)

### Install and Run Frontend

#### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script
python setup_frontend.py
```

The setup script will:
- ✅ Check dependencies
- ✅ Install missing packages
- ✅ Verify RAG system is loaded
- ✅ Start the frontend automatically

#### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements_frontend.txt

# Start the frontend
streamlit run movie_rag_frontend.py
```

## 🎯 Usage

### Chat Interface
1. **Type your question** in the chat input at the bottom
2. **Select a retriever method** from the sidebar
3. **View the AI response** with metadata (retriever used, response time)
4. **Try example queries** from the quick actions panel

### Example Questions
- "What are the best rated movies in the database?"
- "Tell me about the reviews for 'The Shawshank Redemption'"
- "What movies have the highest audience scores?"
- "Show me movies directed by Christopher Nolan"
- "What are the most popular genres?"
- "Tell me about recent blockbuster movies"

### Retriever Methods
- **Naive**: Basic embedding-based retrieval
- **BM25**: Traditional keyword-based retrieval
- **Multi-Query**: Generates multiple queries for better coverage
- **Parent-Document**: Retrieves larger document chunks
- **Contextual-Compression**: Uses Cohere reranking for better relevance
- **Ensemble**: Combines multiple retrievers for optimal results

## 🎨 UI Features

### Movie Theme
- **Gradient Headers**: Red to cyan gradients matching movie aesthetics
- **Chat Bubbles**: Styled message containers with movie icons
- **Interactive Buttons**: Hover effects and smooth animations
- **Status Indicators**: Real-time system status and statistics

### Sidebar Controls
- **Retriever Selection**: Dropdown to choose retrieval method
- **System Status**: Live status of RAG components
- **Conversation Stats**: Query count and response times
- **Clear Conversation**: Reset chat history

### Main Chat Area
- **Message History**: Persistent chat with user and AI messages
- **Metadata Display**: Shows which retriever was used and response time
- **Loading Indicators**: Spinner while AI is processing
- **Error Handling**: Graceful error messages

## 🔧 Technical Details

### Architecture
```
Frontend (Streamlit) → movie_rag_system.py → Notebook Functions
```

### Key Components
- **`movie_rag_frontend.py`**: Main Streamlit application
- **`movie_rag_system.py`**: Bridge module to access notebook functions
- **`setup_frontend.py`**: Automated setup and dependency management
- **`requirements_frontend.txt`**: Python dependencies

### Integration Points
- **Direct Function Calls**: Frontend calls `query_enhanced_agent_with_tracing()`
- **Retriever Switching**: Dynamic retriever selection via sidebar
- **Session State**: Persistent chat history and statistics
- **Error Handling**: Graceful fallbacks for missing components

## 🛠️ Troubleshooting

### "RAG System Not Loaded"
1. Make sure the notebook is running
2. Verify all cells in `movie_reviews_rag_system.ipynb` have been executed
3. Check that the notebook kernel is active

### Missing Dependencies
```bash
# Install all required packages
pip install -r requirements_frontend.txt
```

### Import Errors
```bash
# Check if all modules are available
python movie_rag_system.py
```

### Frontend Won't Start
```bash
# Check Streamlit installation
pip install streamlit

# Try running directly
streamlit run movie_rag_frontend.py
```

## 📊 Performance Tips

### For Best Performance
1. **Use Ensemble Retriever**: Best overall results
2. **Keep Notebook Running**: Ensures RAG system stays loaded
3. **Clear Chat Periodically**: Prevents memory buildup
4. **Monitor Response Times**: Use sidebar statistics

### Expected Response Times
- **Naive Retriever**: ~2-3 seconds
- **BM25 Retriever**: ~1-2 seconds  
- **Multi-Query**: ~3-4 seconds
- **Contextual-Compression**: ~4-5 seconds
- **Ensemble**: ~3-4 seconds

## 🎬 Movie Theme Details

### Color Scheme
- **Primary**: Red (#FF6B6B) to Cyan (#4ECDC4) gradients
- **Secondary**: Purple (#667eea) to Pink (#764ba2) gradients
- **Background**: Light gray (#f0f2f6) for user messages
- **Accent**: White (#fff) for AI messages

### Icons and Emojis
- 🎬 Movie AI responses
- 🎭 User messages
- 🔍 Retriever indicators
- 📊 Statistics and metrics
- ⏱️ Response time indicators

## 🔮 Future Enhancements

### Planned Features
- **Movie Posters**: Display movie images in responses
- **Rating Visualizations**: Charts and graphs for ratings
- **Voice Input**: Speech-to-text for questions
- **Export Chat**: Save conversation history
- **Advanced Filters**: Filter by genre, year, rating
- **Movie Recommendations**: AI-powered suggestions

### Technical Improvements
- **Caching**: Cache frequent queries for faster responses
- **Async Processing**: Non-blocking UI during AI processing
- **WebSocket**: Real-time streaming responses
- **Mobile Optimization**: Better mobile experience

## 📝 License

This frontend is part of the AIE7 certification challenge and uses the same license as the main project.

---

**🎬 Enjoy exploring movies with your AI assistant!** 