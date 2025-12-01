# Automated News Analytic Agent

An AI-powered news analysis agent built with LangGraph and Tavily API that automatically searches, summarizes, and generates comprehensive reports on recent news topics with proper citations.

## Features

- **🤖 Agentic Architecture**: LLM decides when to search, what to search for, and how to analyze (NEW!)
- **Intelligent Query Processing**: Automatically extracts topic, number of articles, and time range from natural language input
- **MCP-Based Architecture**: News search abstracted into a local MCP (Model Context Protocol) server for modularity and extensibility
- **Web Search Integration**: Uses Tavily API for advanced web search with date filtering (via MCP server)
- **AI-Powered Summarization**: Leverages OpenAI models to analyze and summarize news articles
- **Citation Management**: Automatically includes citations with URLs for all sources
- **LangGraph Workflow**: Built on LangGraph for robust agentic orchestration

## Requirements

- Python 3.8+
- OpenAI API key
- Tavily API key

## Installation

1. Clone or navigate to this directory:
```bash
cd /mnt/ssd2/benren/deepfake/code/orchestrator/websearch_agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

### Command Line Interface

Run the agent interactively:
```bash
python news_agent.py
```

The agent will prompt you for:
- **Topic**: What field/subject you want news about (e.g., "AI developments", "climate change", "technology")
- **Number of articles**: How many news articles to search (default: 10)
- **Freshness**: How recent the news should be in days (default: 7 for a week)

### Example Queries

```
Search for 10 news articles about AI developments in the past week
```

```
Find 15 recent news about climate change from the last 3 days
```

```
What happened in the technology field in the past 5 days? Search 20 articles.
```

### Programmatic Usage

You can also use the agent programmatically:

```python
from news_agent import NewsAnalyticAgent

# Initialize the agent
agent = NewsAnalyticAgent()

# Run with user input
result = agent.run("Search for 10 news articles about AI in the past week")

# Access the report
print(result["report"])

# Access individual components
print(result["topic"])
print(result["summary"])
print(result["citations"])
```

## Architecture

The agent uses a modular MCP (Model Context Protocol) architecture with **agentic decision-making**:

- **MCP Server** (`mcp_news_server.py`): Handles all news search operations via Tavily API
- **MCP Client** (`mcp_news_client.py`): Provides a client interface to communicate with the MCP server
- **MCP Tools** (`mcp_tools.py`): LangChain tool wrappers that expose MCP functionality to the LLM
- **News Agent** (`news_agent.py`): Main agent that uses LangGraph's `create_react_agent` for agentic tool calling

### Agentic Mode (Default)

In agentic mode, the LLM:
- **Decides when to search**: Only searches when news information is needed
- **Chooses search parameters**: Determines query, number of results, and date range
- **Analyzes results**: Interprets search results and provides insights
- **Adapts to user needs**: Can perform multiple searches or skip searching if not needed

This architecture provides:
- **Agentic Decision-Making**: LLM autonomously decides when and how to use tools
- **Modularity**: Search functionality is decoupled from the main agent
- **Extensibility**: Easy to add new search sources or modify search logic
- **Testability**: MCP server can be tested independently
- **Reusability**: MCP server can be used by other agents or applications

## Workflow

### Agentic Mode (Default)

The agent uses LangGraph's `create_react_agent` pattern:

1. **User Query**: User asks a question about news
2. **LLM Decision**: LLM analyzes the query and decides if/when to use `search_news_tool`
3. **Tool Execution**: If needed, LLM calls `search_news_tool` with appropriate parameters
4. **Analysis**: LLM analyzes search results and provides insights
5. **Response**: LLM generates a comprehensive response with citations

The LLM has full autonomy to:
- Decide whether to search at all
- Choose search parameters (query, number of results, date range)
- Perform multiple searches if needed
- Skip searching if the query doesn't require it

### Legacy Mode

For backward compatibility, the agent can use a deterministic workflow:

1. **Collect Input**: Extracts topic, number of articles, and freshness from user input
2. **Search News**: Uses MCP client to communicate with MCP server, which searches via Tavily API
3. **Summarize**: Analyzes and summarizes the collected news articles
4. **Generate Report**: Creates a comprehensive report with citations

## MCP Server

The MCP server runs locally and provides the following tools:

- `search_news`: Search for news articles with query, result count, and date range filtering
- `get_article_content`: Get full content of a news article (placeholder for future implementation)

The server is automatically started by the MCP client when needed. You can also run it manually:

```bash
python mcp_news_server.py
# or
bash start_mcp_server.sh
```

## Output Format

The agent generates a markdown report containing:

- **Time Period**: The date range covered
- **Executive Summary**: AI-generated summary of key events and perspectives
- **Citations**: Numbered list of all sources with URLs

Example output:
```markdown
# News Analysis Report: AI Developments

## Time Period
Analysis covers news from the past 7 days.

## Executive Summary
[Comprehensive summary of recent AI news...]

## Citations:

1. Title of Article 1
   URL: https://example.com/article1

2. Title of Article 2
   URL: https://example.com/article2
...
```

## Configuration

You can customize the agent by modifying the `NewsAnalyticAgent` initialization:

```python
# Agentic mode (default) - LLM decides when to search
agent = NewsAnalyticAgent(
    model_name="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo", etc.
    temperature=0.7,  # Adjust creativity (0.0-1.0)
    use_mcp=True,  # Use MCP client (default: True)
    agentic=True  # Use agentic architecture (default: True)
)

# Legacy mode - deterministic workflow
agent = NewsAnalyticAgent(
    model_name="gpt-4o-mini",
    temperature=0.7,
    use_mcp=True,
    agentic=False  # Use legacy deterministic workflow
)
```

### Using MCP Client Directly

You can also use the MCP client independently:

```python
from mcp_news_client import NewsSearchMCPClientSync

# Use as context manager
with NewsSearchMCPClientSync() as client:
    results = client.search_news(
        query="AI developments",
        num_results=10,
        date_range_days=7
    )
    print(results)
```

## API Keys

### Getting a Tavily API Key

1. Visit [Tavily.com](https://tavily.com)
2. Sign up for an account
3. Navigate to your dashboard to get your API key

### Getting an OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key

## Troubleshooting

### "TAVILY_API_KEY not found"
- Make sure you've created a `.env` file with your API keys
- Verify the key is correctly set in the `.env` file

### "No news articles found"
- Try broadening your search topic
- Increase the number of days for freshness
- Check your internet connection

### Rate Limiting
- Tavily and OpenAI have rate limits
- If you hit limits, wait a few minutes and try again
- Consider upgrading your API plan if needed

## License

This project is provided as-is for educational and research purposes.

