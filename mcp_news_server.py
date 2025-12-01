"""
MCP Server for News Search Operations
Provides a standardized interface for news searching via MCP protocol
"""

import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


class SearchNewsParams(BaseModel):
    """Parameters for news search"""
    query: str = Field(description="Search query for news articles")
    num_results: int = Field(default=10, ge=1, le=50, description="Number of results to return")
    date_range_days: Optional[int] = Field(default=None, ge=1, le=365, description="Number of days to look back")
    sources: Optional[List[str]] = Field(default=None, description="Specific news sources to search (optional)")


# Create server instance
server = Server("news-search-server")

# Initialize Tavily search tool
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY not found in environment variables")

search_tool = TavilySearchResults(
    max_results=50,
    search_depth="advanced",
    include_answer=False,
    include_raw_content=True,
    include_images=False
)


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="search_news",
            description="Search for news articles using Tavily API. Supports date filtering and result limiting.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for news articles"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (1-50)",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 10
                    },
                    "date_range_days": {
                        "type": "integer",
                        "description": "Number of days to look back (1-365)",
                        "minimum": 1,
                        "maximum": 365
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific news sources to search (optional)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_article_content",
            description="Get full content of a news article from URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the news article"
                    }
                },
                "required": ["url"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    if name == "search_news":
        return await _search_news(arguments)
    elif name == "get_article_content":
        return await _get_article_content(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _search_news(arguments: Dict[str, Any]) -> List[TextContent]:
    """Execute news search"""
    try:
        params = SearchNewsParams(**arguments)
        
        # Build search query with date constraints if provided
        search_query = params.query
        if params.date_range_days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=params.date_range_days)
            search_query = f"{params.query} news {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        
        # Perform search (Tavily is synchronous, so we run it in executor)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: search_tool.invoke({
                "query": search_query,
                "max_results": params.num_results
            })
        )
        
        # Format results
        formatted_results = []
        for i, result in enumerate(results[:params.num_results], 1):
            if isinstance(result, dict):
                formatted_result = {
                    "number": i,
                    "title": result.get("title", "Untitled"),
                    "url": result.get("url", ""),
                    "content": result.get("content", result.get("snippet", "")),
                    "raw_content": result.get("raw_content")
                }
                formatted_results.append(formatted_result)
        
        # Return as JSON string for MCP
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": True,
                "count": len(formatted_results),
                "results": formatted_results
            }, indent=2)
        )]
        
    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({
                "success": False,
                "error": str(e)
            })
        )]


async def _get_article_content(arguments: Dict[str, Any]) -> List[TextContent]:
    """Get full article content from URL"""
    # For now, this is a placeholder - could integrate with article extraction services
    url = arguments.get("url", "")
    return [TextContent(
        type="text",
        text=f"Article content retrieval for {url} - Feature to be implemented"
    )]


async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
