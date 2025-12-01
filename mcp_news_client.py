"""
MCP Client for News Search Operations
Provides a client interface to connect to the news search MCP server
"""

import json
import asyncio
import subprocess
from typing import List, Dict, Any, Optional
from pathlib import Path


class NewsSearchMCPClientSync:
    """
    Synchronous MCP client for news search operations.
    This client manages the MCP server process and communicates via stdio.
    """
    
    def __init__(self, server_script_path: Optional[str] = None):
        """
        Initialize the MCP client
        
        Args:
            server_script_path: Path to the MCP server script. If None, uses default path.
        """
        if server_script_path is None:
            # Default to the mcp_news_server.py in the same directory
            current_dir = Path(__file__).parent
            server_script_path = str(current_dir / "mcp_news_server.py")
        
        self.server_script_path = server_script_path
        self._process = None
        self._loop = None
    
    def _get_loop(self):
        """Get or create event loop"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop
    
    async def _search_news_async(
        self,
        query: str,
        num_results: int = 10,
        date_range_days: Optional[int] = None,
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Async implementation of news search using MCP protocol
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path]
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                arguments = {
                    "query": query,
                    "num_results": num_results
                }
                
                if date_range_days:
                    arguments["date_range_days"] = date_range_days
                
                if sources:
                    arguments["sources"] = sources
                
                try:
                    result = await session.call_tool("search_news", arguments)
                    
                    # Parse the JSON response
                    if result.content and len(result.content) > 0:
                        response_text = result.content[0].text
                        response_data = json.loads(response_text)
                        
                        if response_data.get("success", False):
                            return response_data.get("results", [])
                        else:
                            raise Exception(f"Search failed: {response_data.get('error', 'Unknown error')}")
                    else:
                        raise Exception("Empty response from MCP server")
                        
                except Exception as e:
                    raise Exception(f"Error calling MCP server: {str(e)}")
    
    def search_news(
        self,
        query: str,
        num_results: int = 10,
        date_range_days: Optional[int] = None,
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for news articles (synchronous interface)
        
        Args:
            query: Search query
            num_results: Number of results to return
            date_range_days: Number of days to look back
            sources: Optional list of specific sources
            
        Returns:
            List of search results as dictionaries
        """
        loop = self._get_loop()
        return loop.run_until_complete(
            self._search_news_async(query, num_results, date_range_days, sources)
        )
    
    def close(self):
        """Close the client and cleanup resources"""
        if self._loop and not self._loop.is_closed():
            try:
                # Cancel any pending tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                # Run until all tasks are cancelled
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            except Exception:
                pass
            finally:
                self._loop.close()
                self._loop = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Alias for backward compatibility
NewsSearchMCPClient = NewsSearchMCPClientSync
