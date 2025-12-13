"""
Research Agents - Multi-Agent HCI Research System

This module provides agent implementations for the multi-agent research system.

Part 1: AUTOGEN AGENTS (Commented Out)
Part 2: LANGGRAPH TEAMS (Active)
"""

# AutoGen Agents (Commented Out)
"""
# TODO: YOUR CODE HERE - AutoGen Agent Implementation (PRESERVED)
# The following AutoGen-based agent implementations are preserved for reference.
# They use the RoundRobinGroupChat pattern with AssistantAgent.

import os
from typing import Dict, Any, List, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
# Import our research tools
from src.tools.web_search import web_search
from src.tools.paper_search import paper_search


def create_model_client(config: Dict[str, Any]) -> OpenAIChatCompletionClient:
    \"\"\"
    Create model client for AutoGen agents.

    Args:
        config: Configuration dictionary from config.yaml

    Returns:
        OpenAIChatCompletionClient configured for the specified provider
    \"\"\"
    model_config = config.get("models", {}).get("default", {})
    provider = model_config.get("provider", "groq")

    # Groq configuration (uses OpenAI-compatible API)
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        return OpenAIChatCompletionClient(
            model=model_config.get("name", "llama-3.3-70b-versatile"),
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model_capabilities={
                "json_output": False,
                "vision": False,
                "function_calling": True,
            }
        )

    # OpenAI configuration
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        return OpenAIChatCompletionClient(
            model=model_config.get("name", "gpt-4o-mini"),
            api_key=api_key,
            base_url=base_url,
        )

    elif provider == "vllm":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        return OpenAIChatCompletionClient(
            model=model_config.get("name", "gpt-4o-mini"),
            api_key=api_key,
            base_url=base_url,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.GPT_4O,
                "structured_output": True,
            },
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_planner_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    \"\"\"
    Create a Planner Agent using AutoGen.

    The planner breaks down research queries into actionable steps.
    It doesn't use tools, but provides strategic direction.

    Args:
        config: Configuration dictionary
        model_client: Model client for the agent

    Returns:
        AutoGen AssistantAgent configured as a planner
    \"\"\"
    agent_config = config.get("agents", {}).get("planner", {})

    # Load system prompt from config or use default
    default_system_message = \"\"\"You are a Research Planner. Your job is to break down research queries into clear, actionable steps.

When given a research query, you should:
1. Identify the key concepts and topics to investigate
2. Determine what types of sources would be most valuable (academic papers, web articles, etc.)
3. Suggest specific search queries for the Researcher
4. Outline how the findings should be synthesized

Provide your plan in a structured format with numbered steps.
Be specific about what information to gather and why it's relevant.\"\"\"

    # Use custom prompt from config if available, otherwise use default
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a task planner. Break down research queries into actionable steps.":
        system_message = custom_prompt
    else:
        system_message = default_system_message

    planner = AssistantAgent(
        name="Planner",
        model_client=model_client,
        description="Breaks down research queries into actionable steps",
        system_message=system_message,
    )

    return planner


def create_researcher_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    \"\"\"
    Create a Researcher Agent using AutoGen.

    The researcher has access to web search and paper search tools.
    It gathers evidence based on the planner's guidance.

    Args:
        config: Configuration dictionary
        model_client: Model client for the agent

    Returns:
        AutoGen AssistantAgent configured as a researcher with tool access
    \"\"\"
    agent_config = config.get("agents", {}).get("researcher", {})

    # Load system prompt from config or use default
    default_system_message = \"\"\"You are a Research Assistant. Your job is to gather high-quality information from academic papers and web sources.

You have access to tools for web search and paper search. When conducting research:
1. Use both web search and paper search for comprehensive coverage
2. Look for recent, high-quality sources
3. Extract key findings, quotes, and data
4. Note all source URLs and citations
5. Gather evidence that directly addresses the research query\"\"\"

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a researcher. Find and collect relevant information from various sources.":
        system_message = custom_prompt
    else:
        system_message = default_system_message

    # Wrap tools in FunctionTool
    web_search_tool = FunctionTool(
        web_search,
        description="Search the web for articles, blog posts, and general information. Returns formatted search results with titles, URLs, and snippets."
    )

    paper_search_tool = FunctionTool(
        paper_search,
        description="Search academic papers on Semantic Scholar. Returns papers with authors, abstracts, citation counts, and URLs. Use year_from parameter to filter recent papers."
    )

    # Create the researcher with tool access
    researcher = AssistantAgent(
        name="Researcher",
        model_client=model_client,
        tools=[web_search_tool, paper_search_tool],
        description="Gathers evidence from web and academic sources using search tools",
        system_message=system_message,
    )

    return researcher


def create_writer_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    \"\"\"
    Create a Writer Agent using AutoGen.

    The writer synthesizes research findings into coherent responses with proper citations.

    Args:
        config: Configuration dictionary
        model_client: Model client for the agent

    Returns:
        AutoGen AssistantAgent configured as a writer
    \"\"\"
    agent_config = config.get("agents", {}).get("writer", {})

    # Load system prompt from config or use default
    default_system_message = \"\"\"You are a Research Writer. Your job is to synthesize research findings into clear, well-organized responses.

When writing:
1. Start with an overview/introduction
2. Present findings in a logical structure
3. Cite sources inline using [Source: Title/Author]
4. Synthesize information from multiple sources
5. Avoid copying text directly - paraphrase and synthesize
6. Include a references section at the end
7. Ensure the response directly answers the original query

Format your response professionally with clear headings, paragraphs, in-text citations, and a References section at the end.\"\"\"

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a writer. Synthesize research findings into a coherent report.":
        system_message = custom_prompt
    else:
        system_message = default_system_message

    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Synthesizes research findings into coherent, well-cited responses",
        system_message=system_message,
    )

    return writer


def create_critic_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    \"\"\"
    Create a Critic Agent using AutoGen.

    The critic evaluates the quality of the research and writing,
    providing feedback for improvement.

    Args:
        config: Configuration dictionary
        model_client: Model client for the agent

    Returns:
        AutoGen AssistantAgent configured as a critic
    \"\"\"
    agent_config = config.get("agents", {}).get("critic", {})

    # Load system prompt from config or use default
    default_system_message = \"\"\"You are a Research Critic. Your job is to evaluate the quality and accuracy of research outputs.

Evaluate the research and writing on these criteria:
1. **Relevance**: Does it answer the original query?
2. **Evidence Quality**: Are sources credible and well-cited?
3. **Completeness**: Are all aspects of the query addressed?
4. **Accuracy**: Are there any factual errors or contradictions?
5. **Clarity**: Is the writing clear and well-organized?

Provide constructive but thorough feedback. End your evaluation with either "TERMINATE" if approved, or suggest specific improvements.\"\"\"

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a critic. Evaluate the quality and accuracy of research findings.":
        system_message = custom_prompt
    else:
        system_message = default_system_message

    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        description="Evaluates research quality and provides feedback",
        system_message=system_message,
    )

    return critic


def create_research_team(config: Dict[str, Any]) -> RoundRobinGroupChat:
    \"\"\"
    Create the research team as a RoundRobinGroupChat.

    Args:
        config: Configuration dictionary

    Returns:
        RoundRobinGroupChat with all agents configured
    \"\"\"
    # Create model client (shared by all agents)
    model_client = create_model_client(config)

    # Create all agents
    planner = create_planner_agent(config, model_client)
    researcher = create_researcher_agent(config, model_client)
    writer = create_writer_agent(config, model_client)
    critic = create_critic_agent(config, model_client)

    # Create multi-condition termination to catch various completion signals
    # This fixes the issue where agents continue after "APPROVED - RESEARCH COMPLETE"
    termination = (
        TextMentionTermination("TERMINATE") |
        TextMentionTermination("APPROVED - RESEARCH COMPLETE") |
        TextMentionTermination("APPROVED") |
        MaxMessageTermination(max_messages=25)  # Safety limit to prevent infinite loops
    )

    # Create team with round-robin ordering
    team = RoundRobinGroupChat(
        participants=[planner, researcher, writer, critic],
        termination_condition=termination,
    )

    return team

# END OF AUTOGEN IMPLEMENTATION
"""

# LangGraph Teams

# TODO: YOUR CODE HERE - LangGraph Team Implementation
# - Create planning team
# - Create research team
# - Create synthesis team

import re
from typing import Dict, Any, List, Callable
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import state types from the merged orchestrator
from src.autogen_orchestrator import ResearchState, TeamOutput
from src.tools.citation_tool import CitationTool


def _extract_search_queries_from_plan(state: ResearchState, original_query: str) -> List[str]:
    """
    Phase 4: Extract refined search queries from planning output.

    Parses the planning team's output to find:
    - Quoted search terms ("HCI accessibility")
    - Terms after "search for:" or "search terms:"
    - Key concepts mentioned in the plan

    Returns list of search queries with original query first.
    """
    queries = [original_query]

    planning_output = state.get("planning_output")
    if not planning_output:
        return queries

    plan_text = ""
    if isinstance(planning_output, dict):
        plan_text = planning_output.get("output", "")
    elif isinstance(planning_output, str):
        plan_text = planning_output

    if not plan_text:
        return queries

    # Pattern 1: Quoted terms in double quotes
    quoted_terms = re.findall(r'"([^"]{3,50})"', plan_text)
    for term in quoted_terms:
        if term.lower() != original_query.lower() and len(term.split()) <= 6:
            queries.append(term)

    # Pattern 2: Terms after "search" keywords
    search_patterns = [
        r'search\s+(?:for|terms?)[:\s]+([^\n\.\,]{5,60})',
        r'query[:\s]+([^\n\.\,]{5,60})',
        r'look\s+(?:for|up|into)[:\s]+([^\n\.\,]{5,60})',
    ]
    for pattern in search_patterns:
        matches = re.findall(pattern, plan_text, re.IGNORECASE)
        for match in matches:
            term = match.strip().strip('"').strip("'")
            if term and term.lower() != original_query.lower():
                queries.append(term)

    # Pattern 3: Extract HCI-related terms from plan
    hci_keywords = re.findall(
        r'\b((?:HCI|usability|accessibility|user\s+experience|UX|UI|'
        r'interaction\s+design|human\s+factors|gestalt|affordance|'
        r'cognitive\s+load|interface|multimodal|VR|AR|XR|voice\s+interface|'
        r'touchscreen|haptic|eye\s+tracking)[^\n\.,]{0,30})',
        plan_text, re.IGNORECASE
    )
    for term in hci_keywords[:3]:  # Limit to top 3
        term = term.strip()
        if len(term) > 5 and term.lower() != original_query.lower():
            queries.append(term)

    # Deduplicate while preserving order (original query first)
    seen = set()
    unique_queries = []
    for q in queries:
        q_normalized = q.lower().strip()
        if q_normalized not in seen and len(q_normalized) > 3:
            seen.add(q_normalized)
            unique_queries.append(q)

    # Return up to 4 queries (original + 3 variations)
    return unique_queries[:4]


def create_planning_team(config: Dict) -> Callable:
    """
    Planning Team Subgraph with Reflexion.

    Nodes: Planner -> Reflector

    Innovation: Chain-of-Thought decomposition with self-critique loop.
    """
    model = ChatOpenAI(
        model=config["models"]["default"]["name"],
        temperature=0.7,
    )

    def planner_node(state: ResearchState) -> Dict:
        """Create initial research plan using Chain-of-Thought reasoning."""
        system_prompt = """You are an expert Research Planner for HCI (Human-Computer Interaction) topics.

## Reasoning Process (Chain-of-Thought)
1. ANALYZE: What is the core question? What sub-topics are involved?
2. DECOMPOSE: Break into 3-5 specific research tasks
3. PRIORITIZE: Order by importance and logical dependency
4. SPECIFY: Define search terms and expected sources for each task

## Memory Context
Build upon any relevant past research findings provided.

## Output Format
Provide a numbered research plan with actionable steps.
For each step, specify:
- The specific question to answer
- Recommended search terms
- Expected source types (academic papers, web articles, etc.)

End with "PLAN COMPLETE"."""

        context = ""
        if state.get("memory_context"):
            context = f"\n\n## Previous Research Context\n{state['memory_context']}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research Query: {state['query']}{context}")
        ]

        response = model.invoke(messages)

        return {
            "planning_output": {
                "team_name": "planning_team",
                "output": response.content,
                "confidence": 0.8,
                "sources": [],
                "reflection": None,
            }
        }

    def planning_reflector_node(state: ResearchState) -> Dict:
        """Self-critique the research plan for completeness and quality."""
        plan = state.get("planning_output", {}).get("output", "")

        system_prompt = """Evaluate this research plan critically:

## Evaluation Criteria
1. COVERAGE: Does it address all aspects of the original query?
2. SPECIFICITY: Are the search terms specific and actionable?
3. LOGICAL ORDER: Is the sequence of steps logical?
4. COMPLETENESS: Are there any missing perspectives or angles?

## Scoring
- If quality score >= 0.7: Output "PLAN APPROVED" with brief positive feedback
- If quality score < 0.7: Provide specific improvements needed

Be concise but thorough in your evaluation."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original Query: {state['query']}\n\nProposed Plan:\n{plan}")
        ]

        response = model.invoke(messages)
        approved = "APPROVED" in response.content.upper()

        return {
            "planning_output": {
                **state.get("planning_output", {}),
                "reflection": response.content,
                "confidence": 0.9 if approved else 0.5,
            }
        }

    # Build the subgraph
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("reflector", planning_reflector_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "reflector")
    graph.add_edge("reflector", END)

    return graph.compile()


def create_research_team(config: Dict) -> Callable:
    """
    Research Team Subgraph with Parallel Tool Execution.

    Innovation: Uses ThreadPoolExecutor for concurrent paper_search and web_search.
    This is a key performance improvement over sequential execution.
    """

    def research_executor_node(state: ResearchState) -> Dict:
        """Execute research tools in parallel using ThreadPoolExecutor.

        Phase 4 Enhancement: Extract refined search queries from planning output
        and execute multiple parallel searches for better coverage.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.tools.paper_search import paper_search
        from src.tools.web_search import web_search

        paper_results = []
        web_results = []
        tool_log = []

        # Phase 4: Extract search queries from planning output
        original_query = state["query"]
        search_queries = _extract_search_queries_from_plan(state, original_query)

        # Log query expansion
        tool_log.append({
            "tool": "query_expansion",
            "original": original_query,
            "expanded": search_queries,
            "count": len(search_queries)
        })

        # Run searches for ALL query variations IN PARALLEL
        # Use more workers to handle multiple queries efficiently
        max_workers = min(len(search_queries) * 2, 8)  # 2 searches per query, max 8

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            # Submit paper searches for all query variations
            for i, query in enumerate(search_queries):
                futures[executor.submit(paper_search, query, 8)] = f"paper_search_{i}"

            # Submit web searches for top 2 queries only (to avoid rate limiting)
            for i, query in enumerate(search_queries[:2]):
                futures[executor.submit(web_search, query)] = f"web_search_{i}"

            for future in as_completed(futures):
                tool_name = futures[future]
                try:
                    result = future.result(timeout=30)
                    if tool_name.startswith("paper_search"):
                        # Parse paper results and MERGE with existing
                        if isinstance(result, str):
                            parsed = _parse_paper_results(result)
                            if not parsed and result.strip():
                                parsed = [{"raw_text": result, "title": "Search Results"}]
                        elif isinstance(result, list):
                            parsed = result
                        else:
                            parsed = [{"raw": str(result)}]

                        # Merge and deduplicate by title
                        existing_titles = {p.get("title", "") for p in paper_results}
                        for paper in parsed:
                            if paper.get("title", "") not in existing_titles:
                                paper_results.append(paper)
                                existing_titles.add(paper.get("title", ""))

                        tool_log.append({
                            "tool": tool_name,
                            "success": True,
                            "new_papers": len(parsed),
                            "total_papers": len(paper_results)
                        })
                    else:
                        # Parse web results and MERGE with existing
                        if isinstance(result, str):
                            parsed = _parse_web_results(result)
                            if not parsed and result.strip():
                                parsed = [{"raw_text": result, "title": "Web Results"}]
                        elif isinstance(result, list):
                            parsed = result
                        else:
                            parsed = [{"raw": str(result)}]

                        # Merge and deduplicate by URL or title
                        existing_urls = {w.get("url", w.get("title", "")) for w in web_results}
                        for web in parsed:
                            url_key = web.get("url", web.get("title", ""))
                            if url_key not in existing_urls:
                                web_results.append(web)
                                existing_urls.add(url_key)

                        tool_log.append({
                            "tool": tool_name,
                            "success": True,
                            "new_web": len(parsed),
                            "total_web": len(web_results)
                        })
                except Exception as e:
                    tool_log.append({
                        "tool": tool_name,
                        "success": False,
                        "error": str(e)
                    })

        return {
            "paper_results": paper_results,
            "web_results": web_results,
            "tool_calls_log": state.get("tool_calls_log", []) + tool_log,
        }

    def research_reflector_node(state: ResearchState) -> Dict:
        """Evaluate research quality and coverage."""
        paper_count = len(state.get("paper_results", []))
        web_count = len(state.get("web_results", []))

        # Calculate confidence based on source coverage
        confidence = 0.0
        reflection_notes = []

        if paper_count >= 5:
            confidence += 0.4
            reflection_notes.append(f"Good academic coverage: {paper_count} papers")
        elif paper_count >= 2:
            confidence += 0.2
            reflection_notes.append(f"Moderate academic coverage: {paper_count} papers")
        else:
            reflection_notes.append(f"Limited academic coverage: {paper_count} papers")

        if web_count >= 3:
            confidence += 0.3
            reflection_notes.append(f"Good web coverage: {web_count} sources")
        elif web_count >= 1:
            confidence += 0.15
            reflection_notes.append(f"Limited web coverage: {web_count} sources")
        else:
            reflection_notes.append("No web sources found")

        # Bonus for having both source types
        if paper_count > 0 and web_count > 0:
            confidence += 0.3
            reflection_notes.append("Multi-source triangulation available")

        # Extract source titles for output
        source_titles = []
        for p in state.get("paper_results", [])[:5]:
            if isinstance(p, dict):
                source_titles.append(p.get("title", "Unknown Paper"))

        return {
            "research_output": {
                "team_name": "research_team",
                "output": f"Found {paper_count} academic papers and {web_count} web sources",
                "confidence": min(1.0, confidence),
                "sources": source_titles,
                "reflection": "; ".join(reflection_notes),
            }
        }

    # Build the subgraph
    graph = StateGraph(ResearchState)
    graph.add_node("executor", research_executor_node)
    graph.add_node("reflector", research_reflector_node)

    graph.set_entry_point("executor")
    graph.add_edge("executor", "reflector")
    graph.add_edge("reflector", END)

    return graph.compile()


def create_synthesis_team(config: Dict) -> Callable:
    """
    Synthesis Team Subgraph with Multi-Perspective Self-Critique.

    Nodes: Writer -> Critic

    Innovation: Three-perspective evaluation (Academic Rigor, User Needs, Completeness)
    """
    model = ChatOpenAI(
        model=config["models"]["default"]["name"],
        temperature=0.7,
    )

    def writer_node(state: ResearchState) -> Dict:
        """Synthesize research findings into a coherent, well-cited response.

        CRITICAL: Uses CitationTool to build proper inline citations and bibliography.
        """
        papers = state.get("paper_results", [])
        web = state.get("web_results", [])

        # Initialize CitationTool for APA formatting
        citation_tool = CitationTool(style="apa")
        source_registry = {}

        # Build source registry with proper citations
        citation_guide = "## AVAILABLE SOURCES WITH CITATION FORMATS\n\n"
        citation_guide += "USE THESE EXACT CITATION FORMATS IN YOUR RESPONSE:\n\n"

        # Process academic papers
        citation_guide += "### Academic Papers:\n"
        for i, p in enumerate(papers[:7], 1):
            if isinstance(p, dict):
                source_id = f"S{i}"
                title = p.get('title', 'Unknown')
                year = p.get('year', 'n.d.')
                authors_raw = p.get('authors', [])

                # Handle various author formats
                if isinstance(authors_raw, list):
                    authors = [{"name": a.get("name", a) if isinstance(a, dict) else str(a)} for a in authors_raw[:3]]
                elif isinstance(authors_raw, str):
                    authors = [{"name": authors_raw}]
                else:
                    authors = [{"name": "Unknown"}]

                # Get first author's last name for inline citation
                first_author = authors[0].get("name", "Unknown") if authors else "Unknown"
                if " " in first_author:
                    last_name = first_author.split()[-1]
                else:
                    last_name = first_author

                # Build inline citation format [Author, Year] or [Author et al., Year]
                if len(authors) > 2:
                    inline_cite = f"[{last_name} et al., {year}]"
                elif len(authors) == 2:
                    second_author = authors[1].get("name", "Unknown").split()[-1]
                    inline_cite = f"[{last_name} & {second_author}, {year}]"
                else:
                    inline_cite = f"[{last_name}, {year}]"

                # Add to citation tool and get full citation
                source_dict = {
                    "type": "paper",
                    "authors": authors,
                    "year": year,
                    "title": title,
                    "venue": p.get("venue", ""),
                    "url": p.get("url", ""),
                }
                citation_tool.add_citation(source_dict)
                full_citation = citation_tool.format_citation(source_dict)

                # Store in registry
                source_registry[source_id] = {
                    "source_id": source_id,
                    "source_type": "paper",
                    "title": title,
                    "authors": [a.get("name", "") for a in authors],
                    "year": year,
                    "venue": p.get("venue"),
                    "url": p.get("url"),
                    "inline_citation": inline_cite,
                    "full_citation": full_citation,
                }

                # Add to citation guide for LLM
                abstract_snippet = (p.get("abstract", "") or "")[:200]
                citation_guide += f"\n**{source_id}**: {title}\n"
                citation_guide += f"   - CITE AS: {inline_cite}\n"
                citation_guide += f"   - Key info: {abstract_snippet}...\n"

        # Process web sources
        citation_guide += "\n### Web Sources:\n"
        for i, w in enumerate(web[:5], 1):
            if isinstance(w, dict):
                source_id = f"W{i}"
                title = w.get('title', 'Unknown')
                url = w.get('url', '')
                snippet = w.get('snippet', '')[:200]
                year = w.get('published_date', '2024')
                if isinstance(year, str) and len(year) > 4:
                    # Extract year from date string
                    year_match = re.search(r'20\d{2}', year)
                    year = year_match.group() if year_match else '2024'

                # For web sources, use site name or abbreviated title
                site_name = w.get('site_name', '')
                if not site_name and url:
                    # Extract domain as site name
                    domain_match = re.search(r'https?://(?:www\.)?([^/]+)', url)
                    site_name = domain_match.group(1) if domain_match else title[:20]

                inline_cite = f"[{site_name or title[:15]}, {year}]"

                source_dict = {
                    "type": "webpage",
                    "title": title,
                    "url": url,
                    "year": year,
                    "site_name": site_name,
                }
                citation_tool.add_citation(source_dict)
                full_citation = citation_tool.format_citation(source_dict)

                source_registry[source_id] = {
                    "source_id": source_id,
                    "source_type": "web",
                    "title": title,
                    "authors": [],
                    "year": year,
                    "venue": None,
                    "url": url,
                    "inline_citation": inline_cite,
                    "full_citation": full_citation,
                }

                citation_guide += f"\n**{source_id}**: {title}\n"
                citation_guide += f"   - CITE AS: {inline_cite}\n"
                citation_guide += f"   - Key info: {snippet}...\n"

        # Generate bibliography
        bibliography = citation_tool.generate_bibliography()
        bibliography_text = "\n\n## References\n\n"
        for i, ref in enumerate(bibliography, 1):
            bibliography_text += f"{i}. {ref}\n"

        system_prompt = """You are an Academic Writer specializing in HCI research synthesis.

## CRITICAL CITATION REQUIREMENTS
You MUST use the EXACT inline citation formats provided in the source list.
Every factual claim MUST have a citation in [Author, Year] format.
DO NOT make up citations - only use the ones provided.

## Writing Guidelines
1. STRUCTURE: Use clear sections (Introduction, Key Findings, Synthesis, Conclusion)
2. CITATIONS: Include [Author, Year] citations for ALL factual claims
3. BALANCE: Present multiple perspectives where relevant
4. CLARITY: Write for an informed but non-expert audience
5. EVIDENCE: Ground all claims in the provided sources

## Required Sections
- **Introduction**: Context and research question (1 paragraph)
- **Key Findings**: Main evidence with proper citations (2-3 paragraphs, minimum 4 citations)
- **Synthesis**: Patterns, connections, and implications (1-2 paragraphs)
- **Conclusion**: Summary and future directions (1 paragraph)

## Citation Rules
- Cite EVERY factual claim
- Use format: [AuthorLastName, Year] or [Author et al., Year]
- Cite multiple sources when relevant: [Smith, 2023; Jones, 2024]
- Reference sources by their content, not by ID numbers

The References section will be added automatically - focus on inline citations."""

        # Phase 7: Multi-Perspective Synthesis
        # Generate drafts from 3 perspectives in parallel, then synthesize
        from concurrent.futures import ThreadPoolExecutor, as_completed

        perspectives = {
            "academic": """You are an ACADEMIC RESEARCHER focused on theoretical rigor.
Emphasize:
- Peer-reviewed sources and academic methodology
- Theoretical frameworks and foundational concepts
- Research gaps and future directions
- Critical analysis of findings""",

            "practitioner": """You are a PRACTITIONER focused on practical applications.
Emphasize:
- Real-world implementation examples
- Industry best practices and case studies
- Actionable recommendations
- Trade-offs and practical considerations""",

            "critical": """You are a CRITICAL ANALYST focused on limitations and nuance.
Emphasize:
- Potential limitations and counterarguments
- Areas of uncertainty or debate
- Alternative perspectives
- Methodological concerns"""
        }

        def generate_perspective_draft(perspective_name: str, perspective_prompt: str) -> str:
            """Generate a draft from a specific perspective."""
            full_prompt = f"""{perspective_prompt}

## CRITICAL CITATION REQUIREMENTS
You MUST use the EXACT inline citation formats provided in the source list.
Every factual claim MUST have a citation in [Author, Year] format.
DO NOT make up citations - only use the ones provided.

## Writing Guidelines
1. Focus on your assigned perspective ({perspective_name})
2. Include [Author, Year] citations for ALL factual claims
3. Write 2-3 paragraphs from this perspective
4. Be specific and evidence-based"""

            msgs = [
                SystemMessage(content=full_prompt),
                HumanMessage(content=f"Query: {state['query']}\n\n{citation_guide}\n\nWrite from the {perspective_name} perspective with proper citations.")
            ]
            return model.invoke(msgs).content

        # Generate all perspectives in parallel
        perspective_drafts = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(generate_perspective_draft, name, prompt): name
                for name, prompt in perspectives.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    perspective_drafts[name] = future.result()
                except Exception:
                    perspective_drafts[name] = ""

        # Synthesize all perspectives into unified response
        synthesis_prompt = f"""You are an Academic Writer synthesizing multiple expert perspectives.

## CRITICAL CITATION REQUIREMENTS (MUST FOLLOW)
You MUST include AT LEAST 10 inline citations in [Author, Year] format throughout your response.
EVERY factual claim MUST be supported by a citation. DO NOT remove any citations from the perspectives.
Use the EXACT citation formats provided in the source list below.
CITATIONS MUST be distributed throughout ALL sections - not bunched in one place.

{citation_guide}

## Your Task
Combine these three expert perspectives into ONE cohesive, well-structured research synthesis.

### Academic Perspective:
{perspective_drafts.get('academic', '')[:1500]}

### Practitioner Perspective:
{perspective_drafts.get('practitioner', '')[:1500]}

### Critical Perspective:
{perspective_drafts.get('critical', '')[:1500]}

## Required Structure
1. **Introduction**: Frame the research question with context (1 paragraph, include 1-2 citations)
2. **Key Findings**: Synthesize main evidence from all perspectives (2-3 paragraphs, include 4-5 citations)
3. **Practical Implications**: Application insights (1 paragraph, include 1-2 citations)
4. **Limitations & Future Directions**: Critical assessment (1 paragraph, include 1-2 citations)
5. **Conclusion**: Summary of key takeaways (1 paragraph)

## Critical Rules
- PRESERVE and ADD inline citations [Author, Year] - minimum 8 total citations
- Use format: [AuthorLastName, Year] or [Author et al., Year]
- Cite multiple sources when relevant: [Smith, 2023; Jones, 2024]
- Resolve contradictions by presenting both views with their citations
- Ensure every major claim has a citation from the source list"""

        # Build explicit citation list for user message
        allowed_citations_list = "## ALLOWED CITATIONS (USE ONLY THESE):\n"
        for sid, info in source_registry.items():
            cite = info.get("inline_citation", "[Unknown]")
            title = info.get("title", "")[:50]
            allowed_citations_list += f"  - {cite} : {title}\n"

        messages = [
            SystemMessage(content=synthesis_prompt),
            HumanMessage(content=f"""Original Query: {state['query']}

{allowed_citations_list}

CRITICAL INSTRUCTION: You MUST use citations from the ALLOWED CITATIONS list above.
DO NOT invent or fabricate any author names. Use ONLY the exact citation formats shown above.
If you cannot find a citation for a claim, do not include that claim.

REQUIRED: Include at least 10 inline citations distributed across ALL sections.
REQUIRED: End with a "## References" section listing all cited sources in APA format.

Synthesize the perspectives above into a unified research response with at least 10 citations from the allowed list.""")
        ]

        response = model.invoke(messages)

        draft = response.content
        if "DRAFT COMPLETE" in draft:
            draft = draft.replace("DRAFT COMPLETE", "").strip()

        # Append the bibliography
        draft = draft + bibliography_text

        return {
            "draft": draft,
            "source_registry": source_registry,
            "perspective_drafts": perspective_drafts,  # Store for debugging
        }

    def critic_node(state: ResearchState) -> Dict:
        """Multi-perspective quality evaluation of the draft."""
        system_prompt = """Evaluate this research response from THREE critical perspectives:

### 1. Academic Rigor (Weight: 35%)
- Are claims properly cited?
- Are sources credible and relevant?
- Is the argumentation logical?

### 2. User Needs (Weight: 35%)
- Does it directly answer the query?
- Is the language clear and accessible?
- Is the structure easy to follow?

### 3. Completeness (Weight: 30%)
- Are all aspects of the query addressed?
- Is coverage balanced across topics?
- Are limitations acknowledged?

## Scoring Instructions
Rate each dimension from 1-5:
1 = Poor, 2 = Below Average, 3 = Average, 4 = Good, 5 = Excellent

## Output Format
Provide scores and brief justification for each perspective.

Final Decision:
- If weighted average >= 3.5: "APPROVED - RESEARCH COMPLETE"
- Otherwise: "NEEDS REVISION: [specific actionable feedback]"
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Original Query: {state['query']}\n\nDraft Response:\n{state.get('draft', '')}")
        ]

        response = model.invoke(messages)
        approved = "APPROVED" in response.content.upper()

        return {
            "critique": response.content,
            "synthesis_output": {
                "team_name": "synthesis_team",
                "output": state.get("draft", ""),
                "confidence": 0.85 if approved else 0.4,
                "sources": [],
                "reflection": response.content,
            }
        }

    # Build the subgraph
    graph = StateGraph(ResearchState)
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)

    graph.set_entry_point("writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("critic", END)

    return graph.compile()


def _parse_paper_results(result_str: str) -> List[Dict]:
    """Parse paper search results from string format.

    Handles the numbered format from paper_search():
    1. Paper Title
       Authors: Smith, Jones
       Year: 2023 | Citations: 45 | Venue: CHI
       Abstract: This is...
       URL: https://...
    """
    papers = []
    if not result_str:
        return papers

    lines = result_str.split("\n")
    current_paper = {}

    for line in lines:
        line_stripped = line.strip()

        if re.match(r"^\d+\.\s+", line_stripped):
            if current_paper:
                papers.append(current_paper)
            title = re.sub(r"^\d+\.\s+", "", line_stripped)
            current_paper = {"title": title}

        elif line_stripped.startswith("Authors:"):
            # Parse authors as a list of dicts for proper citation handling
            authors_str = line_stripped[8:].strip()
            # Split by "," or " et al" and create list of author dicts
            author_names = [a.strip() for a in authors_str.replace(" et al.", "").replace(" et al", "").split(",") if a.strip()]
            current_paper["authors"] = [{"name": name} for name in author_names[:3]]  # Max 3 authors

        elif line_stripped.startswith("Year:"):
            parts = line_stripped.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("Year:"):
                    current_paper["year"] = part[5:].strip()
                elif part.startswith("Citations:"):
                    current_paper["citations"] = part[10:].strip()
                elif part.startswith("Venue:"):
                    current_paper["venue"] = part[6:].strip()

        elif line_stripped.startswith("Abstract:"):
            current_paper["abstract"] = line_stripped[9:].strip()

        elif line_stripped.startswith("URL:"):
            current_paper["url"] = line_stripped[4:].strip()

    if current_paper:
        papers.append(current_paper)

    return papers


def _parse_web_results(result_str: str) -> List[Dict]:
    """Parse web search results from string format.

    Handles the numbered format from web_search():
    1. Title Here
       URL: https://...
       Snippet text here
       Published: date (optional)
    """
    results = []
    if not result_str:
        return results

    lines = result_str.split('\n')
    current_result = {}

    for line in lines:
        line_stripped = line.strip()

        # Match numbered title format: "1. Title" or "2. Another Title"
        if re.match(r'^\d+\.\s+', line_stripped):
            if current_result:
                results.append(current_result)
            title = re.sub(r'^\d+\.\s+', '', line_stripped)
            current_result = {'title': title}

        elif line_stripped.startswith('URL:'):
            current_result['url'] = line_stripped[4:].strip()

        elif line_stripped.startswith('Published:'):
            current_result['published_date'] = line_stripped[10:].strip()

        elif line_stripped and current_result and 'title' in current_result:
            # Any other non-empty line after title is likely a snippet
            if 'snippet' not in current_result:
                current_result['snippet'] = line_stripped
            else:
                current_result['snippet'] += ' ' + line_stripped

    if current_result:
        results.append(current_result)

    return results
