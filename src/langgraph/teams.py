"""
Team Subgraphs for LangGraph Multi-Agent Research System

Implements three specialized teams as nested subgraphs:
1. Planning Team: Query decomposition with Chain-of-Thought reasoning
2. Research Team: Parallel tool execution (paper_search + web_search)
3. Synthesis Team: Writing and self-critique with multi-perspective evaluation
"""

import re
from typing import Dict, Any, List, Callable
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .state import ResearchState, TeamOutput


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
        """Execute research tools in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.tools.paper_search import paper_search
        from src.tools.web_search import web_search

        paper_results = []
        web_results = []
        tool_log = []

        # Extract search query - use planning output if available
        search_query = state["query"]

        # Run both searches IN PARALLEL for efficiency
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(paper_search, search_query, 10): "paper_search",
                executor.submit(web_search, search_query): "web_search",
            }

            for future in as_completed(futures):
                tool_name = futures[future]
                try:
                    result = future.result(timeout=30)
                    if tool_name == "paper_search":
                        # Parse paper results
                        if isinstance(result, str):
                            # Parse the string format from paper_search
                            paper_results = _parse_paper_results(result)
                            # Fallback: if parsing failed but result has content, store raw
                            if not paper_results and result.strip():
                                paper_results = [{"raw_text": result, "title": "Search Results"}]
                        elif isinstance(result, list):
                            paper_results = result
                        else:
                            paper_results = [{"raw": str(result)}]
                    else:
                        # Parse web results
                        if isinstance(result, str):
                            web_results = _parse_web_results(result)
                            # Fallback: if parsing failed but result has content, store raw
                            if not web_results and result.strip():
                                web_results = [{"raw_text": result, "title": "Web Results"}]
                        elif isinstance(result, list):
                            web_results = result
                        else:
                            web_results = [{"raw": str(result)}]

                    tool_log.append({
                        "tool": tool_name,
                        "success": True,
                        "count": len(paper_results) if tool_name == "paper_search" else len(web_results)
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
        """Synthesize research findings into a coherent, well-cited response."""
        papers = state.get("paper_results", [])
        web = state.get("web_results", [])

        # Build evidence section from gathered sources
        evidence = "## Academic Papers\n"
        for i, p in enumerate(papers[:7], 1):
            if isinstance(p, dict):
                title = p.get('title', 'Unknown')
                year = p.get('year', 'n.d.')
                authors = p.get('authors', 'Unknown')
                if isinstance(authors, list):
                    authors = ", ".join(authors[:2]) + (" et al." if len(authors) > 2 else "")
                evidence += f"{i}. {title} ({authors}, {year})\n"
            else:
                evidence += f"{i}. {str(p)[:100]}\n"

        evidence += "\n## Web Sources\n"
        for i, w in enumerate(web[:5], 1):
            if isinstance(w, dict):
                title = w.get('title', 'Unknown')
                snippet = w.get('snippet', '')[:150]
                evidence += f"{i}. {title}: {snippet}...\n"
            else:
                evidence += f"{i}. {str(w)[:100]}\n"

        system_prompt = """You are an Academic Writer specializing in HCI research synthesis.

## Writing Guidelines
1. STRUCTURE: Use clear sections (Introduction, Key Findings, Synthesis, Conclusion)
2. CITATIONS: Include [Author, Year] citations for all claims
3. BALANCE: Present multiple perspectives where relevant
4. CLARITY: Write for an informed but non-expert audience
5. EVIDENCE: Ground all claims in the provided sources

## Required Sections
- **Introduction**: Context and research question
- **Key Findings**: Main evidence with proper citations
- **Synthesis**: Patterns, connections, and implications
- **Conclusion**: Summary and future directions
- **References**: List all cited sources in APA format

End your response with "DRAFT COMPLETE"."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {state['query']}\n\n{evidence}")
        ]

        response = model.invoke(messages)

        draft = response.content
        if draft.endswith("DRAFT COMPLETE"):
            draft = draft[:-14].strip()
        elif "DRAFT COMPLETE" in draft:
            draft = draft.replace("DRAFT COMPLETE", "").strip()

        return {"draft": draft}

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
            current_paper["authors"] = line_stripped[8:].strip()

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
