# 🚀 LangGraph + MCP Server Integration

This project demonstrates how to build a LangGraph application that interacts with your MCP (Model Context Protocol) server tools.

## 📁 Files Overview

### 1. `langgraph_app.py` - Basic Demo
- **Mock MCP tools** for demonstration purposes
- **Simple workflow** with basic parameter extraction
- **Good for learning** LangGraph concepts

### 2. `langgraph_mcp_integration.py` - Real Integration
- **Connects to your actual MCP server** tools
- **Enhanced parameter extraction** using regex patterns
- **Real functionality** with your translation, dice rolling, and web search tools

## 🛠️ Prerequisites

Make sure you have the required dependencies:

```bash
uv sync
```

## 🚀 How to Run

### Option 1: Basic Demo (Mock Tools)
```bash
uv run langgraph_app.py
```

### Option 2: Real MCP Integration
```bash
uv run langgraph_mcp_integration.py
```

## 🌟 Features

### 🔄 **LangGraph Workflow**
1. **Analyze Request** - Classifies user input and extracts parameters
2. **Execute Task** - Calls appropriate MCP tools
3. **Format Output** - Presents results in a user-friendly way

### 🎯 **Supported Tasks**

#### 🌍 **Translation**
- `"translate hello world to russian"`
- `"how do you say good morning in spanish"`
- `"what is thank you in french"`

#### 🎲 **Dice Rolling**
- `"roll 2d6"`
- `"2d20 3 times"`
- `"dice 1d100"`

#### 🔍 **Web Search**
- `"search for python tutorials"`
- `"find information about machine learning"`
- `"look up langgraph examples"`

## 🏗️ Architecture

```
User Input → LangGraph Workflow → MCP Server Tools → Formatted Output
     ↓              ↓                    ↓              ↓
  "translate    analyze_request    translate_text   🌍 Translation
   hello to     → execute_task    → roll_dice      Complete!"
   russian"     → format_output   → web_search
```

## 🔧 Customization

### Adding New Tools
1. **Add tool to your MCP server** (`server.py`)
2. **Update the workflow** in LangGraph app
3. **Add parameter extraction** logic
4. **Test the integration**

### Modifying Workflow
- **Add new nodes** for additional processing steps
- **Modify state** to include new data
- **Change routing logic** for different workflows

## 🧪 Testing

### Interactive Mode
Run the app and try these examples:
```
🤖 You: translate hello world to russian
🤖 You: roll 2d20 3 times
🤖 You: search for langgraph tutorials
🤖 You: quit
```

### Programmatic Usage
```python
from langgraph_mcp_integration import run_workflow

# Run workflow programmatically
result = await run_workflow("translate hello world to spanish")
print(result["final_output"])
```

## 🚨 Troubleshooting

### MCP Tools Not Available
- **Restart Cursor** to refresh MCP connection
- **Check MCP configuration** in `~/.cursor/mcp.json`
- **Verify server.py** has correct tool definitions

### LangGraph Errors
- **Check dependencies** with `uv sync`
- **Verify Python version** (requires 3.13+)
- **Check import statements** in the files

## 🔮 Next Steps

### Advanced Features
- **Multi-turn conversations** with memory
- **Tool chaining** (translate → search → summarize)
- **Error handling** and retry logic
- **Async processing** for multiple tools

### Integration Ideas
- **Web interface** using Streamlit/FastAPI
- **Slack/Discord bot** integration
- **Scheduled tasks** with cron
- **API endpoints** for external access

## 📚 Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [FastMCP Server](https://github.com/jlowin/fastmcp)

## 🤝 Contributing

Feel free to:
- **Add new tools** to the MCP server
- **Enhance the workflow** logic
- **Improve parameter extraction**
- **Add new language support**

---

**Happy building with LangGraph + MCP! 🚀✨** 