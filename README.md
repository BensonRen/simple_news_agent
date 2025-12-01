# Automated News Analytic Agent

An AI-powered news analysis agent built with LangGraph and Tavily API that automatically searches, summarizes, and generates comprehensive reports on recent news topics with proper citations.

## Features

- **Intelligent Query Processing**: Automatically extracts topic, number of articles, and time range from natural language input
- **Web Search Integration**: Uses Tavily API for advanced web search with date filtering
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

## Workflow

The agent follows this LangGraph workflow:

1. **Collect Input**: Extracts topic, number of articles, and freshness from user input
2. **Search News**: Uses Tavily API to search for relevant news articles with date filtering
3. **Summarize**: Analyzes and summarizes the collected news articles
4. **Generate Report**: Creates a comprehensive report with citations

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
agent = NewsAnalyticAgent(
    model_name="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo", etc.
    temperature=0.7  # Adjust creativity (0.0-1.0)
)
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

