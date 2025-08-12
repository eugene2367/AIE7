<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 14: Pre-Work](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150?source=copy_link#247cd547af3d80709683ff380f4cba62)| [Session 14: Deploying Agents to Production](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150) | [Recording!](https://us02web.zoom.us/rec/share/1YepNUK3kqQnYLY8InMfHv84JeiOMyjMRWOZQ9jfjY86dDPvHMhyoz5Zo04w_tn-.91KwoSPyP6K6u0DC)  (@@5J6DVQ)| [Session 14 Slides](https://www.canva.com/design/DAGvVPg7-mw/IRwoSgDXPEqU-PKeIw8zLg/edit?utm_content=DAGvVPg7-mw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 14 Assignment: Production Agents](https://forms.gle/nZ7ugE4W9VsC1zXE8) | [AIE7 Feedback 8/7](https://forms.gle/juo8SF5y5XiojFyC9)

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀
- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned


#### ❓ Question:

What is the purpose of the `chunk_overlap` parameter when using `RecursiveCharacterTextSplitter` to prepare documents for RAG, and what trade-offs arise as you increase or decrease its value?

Answer:

# Purpose of `chunk_overlap` in RecursiveCharacterTextSplitter

The `chunk_overlap` parameter determines how many characters overlap between consecutive text chunks when splitting documents for RAG systems. This overlap helps maintain context continuity across chunk boundaries.

## Trade-offs of Increasing `chunk_overlap`

### Benefits:
- **Better context preservation**: Important information spanning chunk boundaries is less likely to be lost
- **Improved retrieval quality**: Related concepts split across chunks can still be retrieved together
- **Reduced information fragmentation**: Sentences or paragraphs cut off mid-thought are less problematic

### Drawbacks:
- **Higher storage costs**: More duplicate content increases vector database size
- **Increased computational overhead**: More redundant processing during embedding and retrieval
- **Potential redundancy**: Same information may appear in multiple chunks, diluting retrieval relevance

## Trade-offs of Decreasing `chunk_overlap`

### Benefits:
- **Lower storage requirements**: Minimal duplication reduces database size
- **Faster processing**: Less redundant content to process
- **Cleaner chunk boundaries**: More distinct, non-overlapping chunks

### Drawbacks:
- **Context loss**: Important information at chunk boundaries may be lost
- **Poorer retrieval**: Related concepts split across chunks might not be retrieved together
- **Reduced coherence**: Individual chunks may lack sufficient context to be meaningful

## Practical Guidelines

- **Default overlap**: Often 200 characters, balancing context preservation with efficiency
- **Technical documents**: Higher overlap (300-500 chars) beneficial for complex terminology
- **Narrative text**: Lower overlap (100-200 chars) may suffice due to natural context flow
- **Chunk size relationship**: Larger chunks can tolerate lower overlap, smaller chunks benefit from higher overlap

The optimal `chunk_overlap` value depends on your specific use case, document type, and desired balance between retrieval quality and system efficiency.


#### ❓ Question:

Your retriever is configured with `search_kwargs={"k": 5}`. How would adjusting `k` likely affect RAGAS metrics such as Context Precision and Context Recall in practice, and why?

Answer:

# Impact of Adjusting `k` on RAGAS Metrics

The `k` parameter in `search_kwargs={"k": 5}` controls how many top-k most similar documents the retriever returns. This significantly affects RAGAS metrics, particularly Context Precision and Context Recall.

## Context Precision Impact

### Increasing `k` (e.g., k=10, k=20):
- **Likely decreases Context Precision**: As you retrieve more documents, you're more likely to include less relevant or irrelevant context
- **Dilution effect**: The additional documents may have lower similarity scores, reducing the overall relevance of the retrieved context
- **Example**: With k=5, you might get 4 highly relevant docs + 1 moderately relevant. With k=20, you might get 5 relevant + 15 less relevant docs

### Decreasing `k` (e.g., k=3, k=1):
- **Likely increases Context Precision**: Fewer, more focused results typically mean higher average relevance
- **Concentration effect**: Only the most similar documents are returned, maintaining high relevance
- **Risk**: May miss important context that's slightly less similar but still relevant

## Context Recall Impact

### Increasing `k`:
- **Likely increases Context Recall**: More documents mean higher chance of retrieving all relevant information
- **Coverage improvement**: Even if some documents are less relevant, you're more likely to capture all necessary context
- **Example**: A comprehensive answer might require information spread across 8 documents - k=5 would miss 3, but k=10 would capture all

### Decreasing `k`:
- **Likely decreases Context Recall**: Fewer documents mean higher risk of missing relevant context
- **Coverage gaps**: Important information in lower-ranked but still relevant documents gets excluded
- **Example**: With k=3, you might miss crucial context that ranks 4th or 5th in similarity

## Practical Trade-offs

- **k=1-3**: High precision, low recall - good for focused, specific questions
- **k=5-10**: Balanced approach - reasonable precision with decent recall
- **k=15-20**: Lower precision, higher recall - good for comprehensive questions requiring broad context

## Why This Happens

The relationship exists because:
1. **Similarity ranking**: Documents are ranked by similarity, so lower-ranked results are inherently less relevant
2. **Information distribution**: Relevant information is often distributed across multiple documents
3. **Precision-recall trade-off**: This is a classic information retrieval trade-off - you can't maximize both simultaneously

The optimal `k` value depends on your specific use case: whether you prioritize getting the most relevant context (lower k) or ensuring comprehensive coverage (higher k).


#### ❓ Question:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

Answer:

# Comparison of `agent` vs `agent_helpful` Assistants

## Graph Structure Differences

### `agent` (simple_agent):
- **Entry point**: `agent` node
- **Flow**: `agent` → conditional routing → `action` (tools) or `END`
- **Simple loop**: `agent` ↔ `action` until no more tool calls needed
- **Termination**: Ends when no tool calls are requested

### `agent_helpful` (agent_with_helpfulness):
- **Entry point**: `agent` node  
- **Flow**: `agent` → conditional routing → `action` (tools) or `helpfulness`
- **Complex loop**: `agent` ↔ `action` ↔ `helpfulness` evaluation
- **Termination**: Ends when helpfulness check passes or loop limit reached

## Where the Helpfulness Evaluator Fits

The helpfulness evaluator (`helpfulness_node`) sits **after** the agent completes its response and tool execution. It acts as a **post-processing quality gate** that:

1. **Evaluates the final response** against the initial query
2. **Makes a binary decision**: Helpful (Y) or Unhelpful (N)
3. **Controls the execution flow** based on this evaluation

## Routing Conditions

### Route back to agent (`continue`):
- **When**: `helpfulness_decision` returns "continue"
- **Condition**: Helpfulness evaluation returns "N" (unhelpful)
- **Purpose**: Allows the agent to improve its response or try a different approach
- **Example**: Agent gives vague answer → helpfulness check fails → loops back for better response

### Terminate execution (`end`):
- **When**: `helpfulness_decision` returns "end" or `END`
- **Conditions**:
  - Helpfulness evaluation returns "Y" (helpful)
  - Loop limit exceeded (10+ messages) - safety mechanism
- **Purpose**: Stops execution when quality threshold is met or safety limit reached

## Key Architectural Differences

| Aspect | `agent` | `agent_helpful` |
|--------|---------|------------------|
| **Complexity** | Simple tool-using loop | Tool-using + quality evaluation loop |
| **Quality Control** | None | Built-in helpfulness assessment |
| **Loop Control** | Tool-driven | Helpfulness-driven + safety limits |
| **Termination** | When tools complete | When helpfulness criteria met |
| **Safety** | No loop protection | 10-message loop limit |

## Why This Design Matters

The helpfulness evaluator creates a **self-improving loop** where:
- The agent can iterate on its responses until they meet quality standards
- There's a built-in safety mechanism to prevent infinite loops
- The system automatically terminates when the response is deemed satisfactory
- Quality is continuously monitored rather than assumed

This makes `agent_helpful` more robust for production use where response quality and system stability are critical.