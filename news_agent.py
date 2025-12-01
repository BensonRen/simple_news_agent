"""
Automated News Analytic Agent using LangGraph and Tavily API
"""

import os
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, Sequence, List, Optional
from textwrap import dedent
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

load_dotenv()

# Enable LangSmith tracing if API key is available
if os.getenv("LANGSMITH_API_KEY"):
    print("LangSmith API key found, tracing is enabled")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "news-analytic-agent")
else:
    print("No LangSmith API key found, tracing is disabled")


class Citation(BaseModel):
    """Citation model for news article references"""
    number: int = Field(description="Citation number in the report")
    title: str = Field(description="Title of the news article")
    url: str = Field(description="URL of the news article")


class SearchResult(BaseModel):
    """Search result model from Tavily API"""
    title: str = Field(description="Title of the search result")
    url: str = Field(description="URL of the search result")
    content: str = Field(default="", description="Content or snippet from the search result")
    raw_content: Optional[str] = Field(default=None, description="Raw content from the search result")


class UserInputExtraction(BaseModel):
    """Model for extracting structured information from user input"""
    topic: str = Field(description="Topic of news (what field/subject they want news about)")
    num_news: int = Field(
        default=10,
        description="Number of news articles to search",
        ge=1,
        le=50
    )
    freshness_days: int = Field(
        default=7,
        description="How fresh the news should be in days",
        ge=1,
        le=365
    )


class AgentState(TypedDict):
    """State of the news analysis agent"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    topic: str
    num_news: int
    freshness_days: int
    search_results: List[dict]
    summary: str
    citations: List[dict]  # Stored as dict, validated with Citation model
    report: str


class NewsAnalyticAgent:
    """AI Agent for automated news analysis using LangGraph and Tavily"""
    
    def __init__(self, model_name: str = "gpt-5-nano", temperature: float = 0):
        """
        Initialize the News Analytic Agent
        
        Args:
            model_name: OpenAI model to use
            temperature: Temperature for the model
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        # Initialize Tavily search tool
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY not found in environment variables")
        
        self.search_tool = TavilySearchResults(
            max_results=20,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=True,
            include_images=False
        )
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("collect_input", self.collect_user_input)
        workflow.add_node("search_news", self.search_news)
        workflow.add_node("summarize", self.summarize_news)
        workflow.add_node("generate_report", self.generate_report)
        
        # Define the flow
        workflow.set_entry_point("collect_input")
        workflow.add_edge("collect_input", "search_news")
        workflow.add_edge("search_news", "summarize")
        workflow.add_edge("summarize", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    def collect_user_input(self, state: AgentState) -> AgentState:
        """Collect user input: topic, number of news, and freshness"""
        print("\n[Step 1/4] 📝 Extracting information from your request...")
        messages = state.get("messages", [])
        
        # Extract information from the last user message
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                content = last_message.content
                print(f"   Analyzing: '{content}'")
                
                # Check if we already have the required info
                if state.get("topic") and state.get("num_news") and state.get("freshness_days"):
                    print("   ✓ Parameters already extracted, skipping...")
                    return state
                
                # Use LLM with structured output to extract information
                extraction_prompt = dedent(f"""
                    Extract the following information from the user's request:
                    1. Topic of news (what field/subject they want news about)
                    2. Number of news articles to search (default: 10, range: 1-50)
                    3. How fresh the news should be in days (default: 7 for a week, range: 1-365)

                    User request: {content}

                    If any information is missing, use the default values.
                    If the topic is not clear, use the entire user request as the topic.
                    """).strip()
                
                # Use structured output to get validated Pydantic model
                print("   Using AI to extract topic, number of articles, and time range...")
                structured_llm = self.llm.with_structured_output(UserInputExtraction)
                
                try:
                    extracted = structured_llm.invoke([HumanMessage(content=extraction_prompt)])
                    
                    # If topic is empty or invalid, use the whole content as topic
                    topic = extracted.topic.strip() if extracted.topic.strip() else content
                    
                    state["topic"] = topic
                    state["num_news"] = extracted.num_news
                    state["freshness_days"] = extracted.freshness_days
                    
                    print(f"   ✓ Extracted parameters:")
                    print(f"      • Topic: '{topic}'")
                    print(f"      • Number of articles: {extracted.num_news}")
                    print(f"      • Time range: past {extracted.freshness_days} days")
                    
                    # Add confirmation message
                    confirmation = f"I'll search for {extracted.num_news} news articles about '{topic}' from the past {extracted.freshness_days} days."
                    state["messages"] = messages + [AIMessage(content=confirmation)]
                except Exception as e:
                    # Fallback to defaults if structured output fails
                    print(f"   ⚠ Warning: Extraction failed, using defaults ({str(e)})")
                    state["topic"] = content
                    state["num_news"] = 10
                    state["freshness_days"] = 7
                    print(f"   ✓ Using defaults: 10 articles, past 7 days")
                    state["messages"] = messages + [
                        AIMessage(content=f"Using defaults: searching for 10 articles about '{content}' from the past 7 days.")
                    ]
        
        return state
    
    def search_news(self, state: AgentState) -> AgentState:
        """Search for news using Tavily API"""
        print("\n[Step 2/4] 🔍 Searching for news articles...")
        topic = state.get("topic", "")
        num_news = state.get("num_news", 10)
        freshness_days = state.get("freshness_days", 7)
        
        if not topic:
            print("   ✗ Error: No topic provided for search.")
            state["search_results"] = []
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="Error: No topic provided for search.")
            ]
            return state
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=freshness_days)
        
        # Build search query with date constraints
        search_query = f"{topic} news {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        print(f"   Query: '{search_query}'")
        print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"   Requesting up to {num_news} articles...")
        
        try:
            # Perform search
            print("   Connecting to Tavily API...")
            results = self.search_tool.invoke({
                "query": search_query,
                "max_results": num_news
            })
            print(f"   ✓ Received {len(results)} results from Tavily")
            
            # Filter results by date if possible and limit to num_news
            print("   Processing and validating results...")
            filtered_results = []
            for i, result in enumerate(results[:num_news], 1):
                # Check if result has date information
                if isinstance(result, dict):
                    # Validate and structure the result using Pydantic
                    try:
                        search_result = SearchResult(
                            title=result.get("title", "Untitled"),
                            url=result.get("url", ""),
                            content=result.get("content", result.get("snippet", "")),
                            raw_content=result.get("raw_content")
                        )
                        filtered_results.append(search_result.model_dump())
                        if i <= 3:  # Show first 3 titles
                            print(f"      {i}. {search_result.title[:60]}...")
                    except Exception as e:
                        # If validation fails, use the raw dict
                        filtered_results.append(result)
                        if i <= 3:
                            print(f"      {i}. {result.get('title', 'Untitled')[:60]}...")
            
            state["search_results"] = filtered_results
            
            print(f"   ✓ Successfully processed {len(filtered_results)} articles")
            
            # Add status message
            status_msg = f"Found {len(filtered_results)} news articles about '{topic}'."
            state["messages"] = state.get("messages", []) + [AIMessage(content=status_msg)]
            
        except Exception as e:
            print(f"   ✗ Error during search: {str(e)}")
            error_msg = f"Error during search: {str(e)}"
            state["search_results"] = []
            state["messages"] = state.get("messages", []) + [AIMessage(content=error_msg)]
        
        return state
    
    def summarize_news(self, state: AgentState) -> AgentState:
        """Summarize the collected news articles"""
        print("\n[Step 3/4] 📊 Analyzing and summarizing news articles...")
        search_results = state.get("search_results", [])
        topic = state.get("topic", "")
        
        if not search_results:
            print("   ✗ No news articles found to summarize.")
            state["summary"] = "No news articles found to summarize."
            return state
        
        print(f"   Processing {len(search_results)} articles about '{topic}'...")
        
        # Format search results for summarization
        articles_text = ""
        citations = []
        
        for i, result in enumerate(search_results, 1):
            # Convert dict to SearchResult if needed
            if isinstance(result, dict):
                try:
                    search_result = SearchResult(**result)
                except Exception:
                    # Fallback to dict access if validation fails
                    search_result = SearchResult(
                        title=result.get("title", "Untitled"),
                        url=result.get("url", ""),
                        content=result.get("content", result.get("raw_content", ""))
                    )
                
                title = search_result.title
                content = search_result.content or search_result.raw_content or ""
                url = search_result.url
                
                articles_text += f"\n\nArticle {i}:\nTitle: {title}\nContent: {content[:1000]}...\n"
                
                # Create Citation using Pydantic model and convert to dict for state
                citation = Citation(
                    number=i,
                    title=title,
                    url=url
                )
                citations.append(citation.model_dump())
        
        print("   Preparing articles for AI analysis...")
        print(f"   ✓ Extracted {len(citations)} citations")
        
        # Create summarization prompt
        summarize_prompt = dedent(f"""
            You are a news analyst. Summarize the following news articles about '{topic}'.

            Focus on:
            1. Key events and developments
            2. Main perspectives and viewpoints
            3. Important trends or patterns
            4. Significant implications

            News Articles:
            {articles_text}

            Provide a comprehensive summary that captures the essence of what happened in this field recently.
            """).strip()
        
        try:
            print("   Sending articles to AI for analysis...")
            response = self.llm.invoke([HumanMessage(content=summarize_prompt)])
            summary = response.content
            state["summary"] = summary
            state["citations"] = citations
            
            print("   ✓ AI analysis complete")
            print(f"   Summary length: {len(summary)} characters")
            
            # Add status message
            state["messages"] = state.get("messages", []) + [
                AIMessage(content="News articles have been analyzed and summarized.")
            ]
        except Exception as e:
            print(f"   ✗ Error during summarization: {str(e)}")
            state["summary"] = f"Error during summarization: {str(e)}"
        
        return state
    
    def generate_report(self, state: AgentState) -> AgentState:
        """Generate final report with citations"""
        print("\n[Step 4/4] 📄 Generating final report...")
        topic = state.get("topic", "")
        summary = state.get("summary", "")
        citations = state.get("citations", [])
        freshness_days = state.get("freshness_days", 7)
        
        print(f"   Topic: '{topic}'")
        print(f"   Time period: past {freshness_days} days")
        print(f"   Citations: {len(citations)} sources")
        
        # Format citations - convert dicts to Citation models for validation
        print("   Formatting citations...")
        citations_text = "\n\n## Citations:\n\n"
        for citation_dict in citations:
            # Validate and convert to Citation model
            try:
                citation = Citation(**citation_dict) if isinstance(citation_dict, dict) else citation_dict
                if isinstance(citation, Citation):
                    citations_text += f"{citation.number}. {citation.title}\n   URL: {citation.url}\n\n"
                else:
                    # Fallback for dict
                    citations_text += f"{citation_dict.get('number', '')}. {citation_dict.get('title', '')}\n   URL: {citation_dict.get('url', '')}\n\n"
            except Exception:
                # Fallback to dict access if validation fails
                citations_text += f"{citation_dict.get('number', '')}. {citation_dict.get('title', '')}\n   URL: {citation_dict.get('url', '')}\n\n"
        
        # Generate report
        print("   Assembling report sections...")
        report = dedent(f"""
            # News Analysis Report: {topic}

            ## Time Period
            Analysis covers news from the past {freshness_days} days.

            ## Executive Summary
            {summary}

            {citations_text}

            ---
            *Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
            """).strip()
        
        state["report"] = report
        
        print(f"   ✓ Report generated successfully!")
        print(f"   Report size: {len(report)} characters")
        
        # Add final message
        state["messages"] = state.get("messages", []) + [
            AIMessage(content=f"Report generated successfully!\n\n{report}")
        ]
        
        return state
    
    def run(self, user_input: str) -> dict:
        """
        Run the agent with user input
        
        Args:
            user_input: User's query about news topic
            
        Returns:
            Final state with report
        """
        print("\n" + "=" * 60)
        print("🚀 Starting News Analysis Workflow")
        print("=" * 60)
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "topic": "",
            "num_news": 10,
            "freshness_days": 7,
            "search_results": [],
            "summary": "",
            "citations": [],
            "report": ""
        }
        
        print(f"User input: '{user_input}'")
        print("\nExecuting workflow steps...")
        
        final_state = self.graph.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("✅ Workflow completed successfully!")
        print("=" * 60)
        
        return final_state


def main():
    """Main function to run the news analytic agent"""
    print("=" * 60)
    print("Automated News Analytic Agent")
    print("=" * 60)
    print("\nThis agent will help you analyze recent news on any topic.")
    print("You can specify:")
    print("  - The topic/field you're interested in")
    print("  - Number of news articles to search (default: 10)")
    print("  - How fresh the news should be in days (default: 7)")
    print("\nExample: 'Search for 10 news articles about AI developments in the past week'")
    print("=" * 60)

    save_folder = 'generated_reports/'
    
    # Initialize agent
    try:
        agent = NewsAnalyticAgent()
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nPlease set TAVILY_API_KEY and OPENAI_API_KEY in your .env file")
        return
    
    # Get user input
    print("\nEnter your news query:")
    user_input = input("> ")
    
    if not user_input.strip():
        print("No input provided. Exiting.")
        return
    
    print("\nProcessing your request...\n")
    
    # Run the agent
    try:
        result = agent.run(user_input)
        
        # Display the report
        print("\n" + "=" * 60)
        print("FINAL REPORT")
        print("=" * 60)
        print(result.get("report", "No report generated."))
        print("=" * 60)
        
        # Optionally save to file
        save = input("\nSave report to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"news_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(os.path.join(save_folder, filename), 'w', encoding='utf-8') as f:
                f.write(result.get("report", ""))
            print(f"Report saved to {filename}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

