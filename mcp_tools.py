"""
LangChain tool wrappers for MCP news search tools
These tools expose MCP functionality to the LLM for agentic decision-making
"""

import json
from typing import Optional, List
from langchain_core.tools import tool
from mcp_news_client import NewsSearchMCPClientSync


@tool
def search_news_tool(
    query: str,
    num_results: int = 10,
    date_range_days: int = 7
) -> str:
    """Search for recent news articles on a given topic using the MCP news search server.
    
    Use this tool when you need to find current news articles about a specific topic.
    The tool will search for articles from the specified number of days back.
    
    Args:
        query: The topic or subject to search for (e.g., "artificial intelligence", "climate change", "technology")
        num_results: Number of articles to return (default: 10, range: 1-50)
        date_range_days: How many days back to search (default: 7, range: 1-365)
    
    Returns:
        A JSON string containing an array of search results. Each result has:
        - number: Article number
        - title: Article title
        - url: Article URL
        - content: Article snippet/content
        - raw_content: Full article content if available
    
    Example:
        search_news_tool("AI developments", num_results=5, date_range_days=3)
    """
    try:
        with NewsSearchMCPClientSync() as client:
            results = client.search_news(
                query=query,
                num_results=num_results,
                date_range_days=date_range_days
            )
            # Return formatted JSON for the LLM
            return json.dumps({
                "success": True,
                "count": len(results),
                "results": results
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def get_mcp_tools() -> List:
    """Get all available MCP tools as LangChain tools"""
    return [search_news_tool]

