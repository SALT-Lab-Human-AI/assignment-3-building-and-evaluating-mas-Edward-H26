"""
Orchestrator - Multi-Agent HCI Research System

Part 1: AUTOGEN IMPLEMENTATION (Commented Out)
Part 2: LANGGRAPH IMPLEMENTATION (Active)
"""

# ============================================================
# PART 1: AUTOGEN IMPLEMENTATION (COMMENTED OUT)
# Original AutoGen-based orchestrator using RoundRobinGroupChat
# ============================================================
"""
# TODO: YOUR CODE HERE - AutoGen Orchestrator Implementation
# - Create AutoGen agents with specialized prompts
# - Implement RoundRobinGroupChat coordination
# - Handle memory and tool integration

# AutoGen-Based Orchestrator
#
# This orchestrator uses AutoGen's RoundRobinGroupChat to coordinate multiple agents
# in a research workflow.
#
# Workflow:
# 1. Planner: Breaks down the query into research steps
# 2. Researcher: Gathers evidence using web and paper search tools
# 3. Writer: Synthesizes findings into a coherent response
# 4. Critic: Evaluates quality and provides feedback

import logging
import asyncio
import re
from typing import Dict, Any, List, Optional
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage

from src.agents.autogen_agents import create_research_team
from src.agents.memory import ResearchMemory


class AutoGenOrchestrator:
    # Orchestrates multi-agent research using AutoGen's RoundRobinGroupChat.
    #
    # This orchestrator manages a team of specialized agents that work together
    # to answer research queries. It uses AutoGen's built-in conversation
    # management and tool execution capabilities.

    def __init__(self, config: Dict[str, Any]):
        # Initialize the AutoGen orchestrator.
        #
        # Args:
        #     config: Configuration dictionary from config.yaml
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")

        # Initialize multi-agent memory system
        self.logger.info("Initializing memory system...")
        self.memory = ResearchMemory(max_findings=20, max_context=10)

        # Create the research team
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)

        self.logger.info("Research team created successfully")

        # Workflow trace for debugging and UI display
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        # Process a research query through the multi-agent system.
        #
        # Args:
        #     query: The research question to answer
        #     max_rounds: Maximum number of conversation rounds
        #
        # Returns:
        #     Dictionary containing:
        #     - query: Original query
        #     - response: Final synthesized response
        #     - conversation_history: Full conversation between agents
        #     - metadata: Additional information about the process
        self.logger.info(f"Processing query: {query}")

        # Get relevant past context from memory
        past_context = self.memory.get_relevant_context(query, k=3)
        if past_context:
            self.logger.info("Found relevant past research in memory")

        try:
            # Run the async query processing with memory context
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._process_query_async(query, max_rounds, past_context)
                    ).result()
            else:
                result = loop.run_until_complete(
                    self._process_query_async(query, max_rounds, past_context)
                )

            # Store findings in memory for future context
            self._store_in_memory(query, result)

            self.logger.info("Query processing complete")
            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }

    async def _process_query_async(
        self, query: str, max_rounds: int = 20, past_context: str = ""
    ) -> Dict[str, Any]:
        # Async implementation of query processing.
        #
        # Args:
        #     query: The research question to answer
        #     max_rounds: Maximum number of conversation rounds
        #     past_context: Relevant context from previous research queries
        #
        # Returns:
        #     Dictionary containing results

        # Get related citations from memory
        related_citations = self.memory.get_related_citations(query, k=5)

        # Create task message with enhanced memory context
        context_section = ""
        if past_context or related_citations:
            context_section = "\\n## Memory Context"

            if past_context:
                context_section += f'''
### Relevant Past Research
The following context from previous research may be helpful:

{past_context}
'''

            if related_citations:
                citation_list = "\\n".join([
                    f"- {c['title']} ({c.get('year', 'n.d.')})"
                    for c in related_citations
                ])
                context_section += f'''
### Previously Found Relevant Sources
{citation_list}
'''

            context_section += "\\nPlease build upon this knowledge where relevant and avoid redundant research.\\n"

        task_message = f'''Research Query: {query}
{context_section}
Please work together to answer this query comprehensively:
1. Planner: Create a research plan
2. Researcher: Gather evidence from web and academic sources
3. Writer: Synthesize findings into a well-cited response
4. Critic: Evaluate the quality and provide feedback'''

        # Run the team
        result = await self.team.run(task=task_message)

        # Extract conversation history
        messages = []
        async for message in result.messages:
            msg_dict = {
                "source": message.source,
                "content": message.content if hasattr(message, 'content') else str(message),
            }
            messages.append(msg_dict)

        # Extract final response
        final_response = ""
        if messages:
            # Get the last message from Writer or Critic
            for msg in reversed(messages):
                if msg.get("source") in ["Writer", "Critic"]:
                    final_response = msg.get("content", "")
                    break

        # If no response found, use the last message
        if not final_response and messages:
            final_response = messages[-1].get("content", "")

        return self._extract_results(query, messages, final_response)

    def _extract_results(self, query: str, messages: List[Dict[str, Any]], final_response: str = "") -> Dict[str, Any]:
        # Extract structured results from the conversation history.
        #
        # Args:
        #     query: Original query
        #     messages: List of conversation messages
        #     final_response: Final response from the team
        #
        # Returns:
        #     Structured result dictionary

        # Extract components from conversation
        research_findings = []
        plan = ""
        critique = ""

        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")

            if source == "Planner" and not plan:
                plan = content

            elif source == "Researcher":
                research_findings.append(content)

            elif source == "Critic":
                critique = content

        # Count sources mentioned in research
        num_sources = 0
        for finding in research_findings:
            # Rough count of sources based on numbered results
            num_sources += finding.count("\\n1.") + finding.count("\\n2.") + finding.count("\\n3.")

        # Clean up final response
        if final_response:
            final_response = final_response.replace("TERMINATE", "").strip()

        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(num_sources, 1),  # At least 1
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set([msg.get("source", "") for msg in messages])),
            }
        }

    def _store_in_memory(self, query: str, result: Dict[str, Any]) -> None:
        # Store query results in memory for future context.
        #
        # Args:
        #     query: The original research query
        #     result: The result dictionary from processing
        if "error" in result:
            return  # Don't store failed queries

        response = result.get("response", "")
        sources = self._extract_sources(result)

        # Store the finding in memory
        self.memory.add_finding(query, response, sources)

        # Store citations for deduplication
        for source in sources:
            self.memory.add_citation(source, {"url": source, "query": query})

        # Update context window with key messages
        for msg in result.get("conversation_history", [])[-5:]:
            self.memory.update_context_window(
                f"{msg.get('source', 'Unknown')}: {msg.get('content', '')[:200]}"
            )

        self.logger.info(f"Stored finding in memory. Total findings: {len(self.memory.findings)}")

    def _extract_sources(self, result: Dict[str, Any]) -> List[str]:
        # Extract source URLs from conversation history.
        #
        # Args:
        #     result: The result dictionary from processing
        #
        # Returns:
        #     List of source URLs found in the conversation
        sources = []
        url_pattern = re.compile(r'https?://[^\\s<>"{}|\\\\^`\\[\\]]+')

        for msg in result.get("conversation_history", []):
            content = msg.get("content", "")
            urls = url_pattern.findall(content)
            for url in urls:
                # Clean up URL (remove trailing punctuation)
                url = url.rstrip('.,;:\\'")]}')
                if url not in sources:
                    sources.append(url)

        return sources[:20]  # Limit to 20 sources

    def get_memory_statistics(self) -> Dict[str, Any]:
        # Get statistics about the memory system.
        #
        # Returns:
        #     Dictionary with memory statistics
        return self.memory.get_statistics()

    def clear_memory(self) -> None:
        # Clear all stored memory.
        self.memory.clear()
        self.logger.info("Memory cleared")

    def save_memory(self, filepath: str) -> None:
        # Save memory state to a file.
        #
        # Args:
        #     filepath: Path to save the memory state
        self.memory.save_to_file(filepath)
        self.logger.info(f"Memory saved to {filepath}")

    def load_memory(self, filepath: str) -> bool:
        # Load memory state from a file.
        #
        # Args:
        #     filepath: Path to load the memory state from
        #
        # Returns:
        #     True if loaded successfully, False otherwise
        success = self.memory.load_from_file(filepath)
        if success:
            self.logger.info(f"Memory loaded from {filepath}")
        else:
            self.logger.warning(f"Failed to load memory from {filepath}")
        return success

    def get_agent_descriptions(self) -> Dict[str, str]:
        # Get descriptions of all agents.
        #
        # Returns:
        #     Dictionary mapping agent names to their descriptions
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Gathers evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }

    def visualize_workflow(self) -> str:
        # Generate a text visualization of the workflow.
        #
        # Returns:
        #     String representation of the workflow
        workflow = '''
AutoGen Research Workflow:

1. User Query
   ↓
2. Planner
   - Analyzes query
   - Creates research plan
   - Identifies key topics
   ↓
3. Researcher (with tools)
   - Uses web_search() tool
   - Uses paper_search() tool
   - Gathers evidence
   - Collects citations
   ↓
4. Writer
   - Synthesizes findings
   - Creates structured response
   - Adds citations
   ↓
5. Critic
   - Evaluates quality
   - Checks completeness
   - Provides feedback
   ↓
6. Decision Point
   - If APPROVED → Final Response
   - If NEEDS REVISION → Back to Writer
        '''
        return workflow


def demonstrate_usage():
    # Demonstrate how to use the AutoGen orchestrator.
    #
    # This function shows a simple example of using the orchestrator.
    import yaml
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create orchestrator
    orchestrator = AutoGenOrchestrator(config)

    # Print workflow visualization
    print(orchestrator.visualize_workflow())

    # Example query
    query = "What are the latest trends in human-computer interaction research?"

    print(f"\\nProcessing query: {query}\\n")
    print("=" * 70)

    # Process query
    result = orchestrator.process_query(query)

    # Display results
    print("\\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\\nQuery: {result['query']}")
    print(f"\\nResponse:\\n{result['response']}")
    print(f"\\nMetadata:")
    print(f"  - Messages exchanged: {result['metadata']['num_messages']}")
    print(f"  - Sources gathered: {result['metadata']['num_sources']}")
    print(f"  - Agents involved: {', '.join(result['metadata']['agents_involved'])}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    demonstrate_usage()
"""

# ============================================================
# PART 2: LANGGRAPH IMPLEMENTATION (ACTIVE)
# LangGraph-based orchestrator with StateGraph architecture
# ============================================================

# TODO: YOUR CODE HERE - LangGraph Orchestrator Implementation
# - Implement state management
# - Create supervisor graph
# - Handle workflow execution

# ----- STATE DEFINITIONS (from src/langgraph/state.py) -----

from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
from langgraph.graph import add_messages


class ReflexionMemory(TypedDict):
    """Memory of past agent reflections for learning."""
    past_failures: List[Dict[str, str]]
    successful_patterns: List[Dict[str, str]]
    tool_effectiveness: Dict[str, float]


class TeamOutput(TypedDict):
    """Output from a specialized team."""
    team_name: str
    output: str
    confidence: float
    sources: List[str]
    reflection: Optional[str]


class SourceRegistry(TypedDict):
    """Registry of sources with citation info for proper attribution."""
    source_id: str  # S1, S2, W1, W2, etc.
    source_type: str  # "paper" or "web"
    title: str
    authors: List[str]
    year: Optional[int]
    venue: Optional[str]
    url: Optional[str]
    inline_citation: str  # [Author, Year] format
    full_citation: str  # Full APA citation


class ResearchState(TypedDict):
    """Comprehensive state for the research workflow."""

    query: str
    query_type: Literal["theoretical", "empirical", "trend", "comparison"]

    reflexion_memory: ReflexionMemory
    memory_context: str
    related_citations: List[Dict]

    planning_output: Optional[TeamOutput]
    research_output: Optional[TeamOutput]
    synthesis_output: Optional[TeamOutput]

    paper_results: List[Dict]
    web_results: List[Dict]
    tool_calls_log: List[Dict]
    source_registry: Dict[str, SourceRegistry]  # Maps source_id to citation info

    draft: str
    critique: str
    final_response: str

    safety_layer1_result: Dict
    safety_layer2_result: Dict
    safety_layer3_result: Dict
    is_safe: bool

    human_feedback_requested: bool
    human_feedback: Optional[str]
    human_approved: Optional[bool]

    llm_judge_score: float
    reflexion_score: float
    human_score: Optional[float]
    triangulated_decision: Literal["approved", "revision_needed", "pending_human"]

    messages: Annotated[List[Any], add_messages]
    current_team: Optional[str]
    iteration: int
    max_iterations: int
    status: Literal["planning", "researching", "synthesizing", "reviewing", "human_review", "approved", "blocked"]


def create_initial_state(query: str, max_iterations: int = 5) -> ResearchState:
    """Create initial state for a new research query."""
    return {
        "query": query,
        "query_type": "theoretical",
        "reflexion_memory": {
            "past_failures": [],
            "successful_patterns": [],
            "tool_effectiveness": {},
        },
        "memory_context": "",
        "related_citations": [],
        "planning_output": None,
        "research_output": None,
        "synthesis_output": None,
        "paper_results": [],
        "web_results": [],
        "tool_calls_log": [],
        "source_registry": {},
        "draft": "",
        "critique": "",
        "final_response": "",
        "safety_layer1_result": {},
        "safety_layer2_result": {},
        "safety_layer3_result": {},
        "is_safe": True,
        "human_feedback_requested": False,
        "human_feedback": None,
        "human_approved": None,
        "llm_judge_score": 0.0,
        "reflexion_score": 0.0,
        "human_score": None,
        "triangulated_decision": "pending_human",
        "messages": [],
        "current_team": None,
        "iteration": 0,
        "max_iterations": max_iterations,
        "status": "planning",
    }


# ----- REFLEXION ENGINE (from src/langgraph/reflexion.py) -----

from datetime import datetime
import json
import os
import re


class ReflexionEngine:
    """
    Reflexion-based self-correction engine.

    Maintains memory of past successes/failures and applies learned
    lessons to improve future performance.
    """

    def __init__(self, memory: Optional[Dict] = None):
        self.memory = memory or {
            "past_failures": [],
            "successful_patterns": [],
            "tool_effectiveness": {},
        }

    def evaluate(
        self,
        query: str,
        draft: str,
        sources: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate output quality and generate improvement feedback.

        Returns:
            {
                "score": float (0-1),
                "feedback": str,
                "lesson": str (for memory),
                "issues": List[str]
            }
        """
        issues = []
        score = 0.0

        if not draft or len(draft) < 100:
            issues.append("Response too brief")
            score -= 0.3
        else:
            score += 0.3

        if not sources:
            issues.append("No sources provided")
            score -= 0.2
        elif len(sources) < 3:
            issues.append("Insufficient sources (< 3)")
            score -= 0.1
        else:
            score += 0.3

        # PROPER citation detection using regex patterns
        citation_analysis = self._analyze_citations(draft, sources)
        citation_count = citation_analysis["citation_count"]
        verified_count = citation_analysis["verified_count"]

        if citation_count == 0:
            issues.append("No inline citations found")
            score -= 0.3
        elif citation_count < 3:
            issues.append(f"Insufficient citations ({citation_count} found, minimum 3 expected)")
            score -= 0.1
        else:
            # Good citation count
            score += 0.2
            # Bonus for verified citations (matching actual sources)
            if verified_count >= 3:
                score += 0.1

        query_words = set(query.lower().split())
        draft_words = set(draft.lower().split())
        overlap = len(query_words & draft_words)
        if overlap < len(query_words) * 0.3:
            issues.append("Response may not address query directly")
            score -= 0.1
        else:
            score += 0.2

        past_lessons = self._get_relevant_lessons(query)
        if past_lessons:
            for lesson in past_lessons:
                if lesson.get("error_type", "") in str(issues):
                    issues.append(f"Repeated past error: {lesson.get('lesson', '')}")
                    score -= 0.1

        score = max(0.0, min(1.0, score + 0.5))

        feedback = self._generate_feedback(issues, score)
        lesson = self._extract_lesson(query, issues, score)

        return {
            "score": score,
            "feedback": feedback,
            "lesson": lesson,
            "issues": issues,
        }

    def update_memory(
        self,
        query: str,
        success: bool,
        lesson: str
    ) -> Dict[str, Any]:
        """Update reflexion memory with new insights."""
        entry = {
            "query": query[:100],
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "lesson": lesson,
            "error_type": "quality" if not success else "none",
        }

        if success:
            self.memory["successful_patterns"].append(entry)
            if len(self.memory["successful_patterns"]) > 20:
                self.memory["successful_patterns"] = self.memory["successful_patterns"][-20:]
        else:
            self.memory["past_failures"].append(entry)
            if len(self.memory["past_failures"]) > 20:
                self.memory["past_failures"] = self.memory["past_failures"][-20:]

        return self.memory

    def update_tool_effectiveness(self, tool: str, success: bool) -> None:
        """Track tool effectiveness for adaptive selection."""
        if tool not in self.memory["tool_effectiveness"]:
            self.memory["tool_effectiveness"][tool] = {"total": 0, "success": 0}

        self.memory["tool_effectiveness"][tool]["total"] += 1
        if success:
            self.memory["tool_effectiveness"][tool]["success"] += 1

    def get_tool_score(self, tool: str) -> float:
        """Get effectiveness score for a tool."""
        if tool not in self.memory["tool_effectiveness"]:
            return 0.5
        data = self.memory["tool_effectiveness"][tool]
        if data["total"] == 0:
            return 0.5
        return data["success"] / data["total"]

    def _get_relevant_lessons(self, query: str) -> List[Dict]:
        """Find past lessons relevant to current query."""
        query_words = set(query.lower().split())
        relevant = []

        for failure in self.memory.get("past_failures", []):
            past_words = set(failure.get("query", "").lower().split())
            overlap = len(query_words & past_words)
            if overlap >= 2:
                relevant.append(failure)

        return relevant[:3]

    def _generate_feedback(self, issues: List[str], score: float) -> str:
        """Generate actionable feedback."""
        if score >= 0.7:
            return "Output quality is acceptable. Minor improvements possible."
        elif score >= 0.5:
            feedback = "Output needs improvement:\n"
            for i, issue in enumerate(issues, 1):
                feedback += f"{i}. {issue}\n"
            return feedback
        else:
            return f"Output quality is poor. Critical issues: {', '.join(issues)}"

    def _extract_lesson(self, query: str, issues: List[str], score: float) -> str:
        """Extract a lesson for future reference."""
        if score >= 0.7:
            return f"Successful approach for query type: {query[:30]}..."
        elif issues:
            return f"Avoid: {issues[0]} for queries like: {query[:30]}..."
        return ""

    def _analyze_citations(self, draft: str, sources: List[Dict]) -> Dict[str, Any]:
        """
        Analyze citations in the draft using proper regex patterns.

        Detects:
        - [Author, Year] format: [Smith, 2023]
        - [Author et al., Year]: [Smith et al., 2023]
        - [Author & Author, Year]: [Smith & Jones, 2023]
        - Multiple citations: [Smith, 2023; Jones, 2024]

        Args:
            draft: The text to analyze
            sources: List of source dictionaries to verify against

        Returns:
            {
                "citation_count": int,
                "verified_count": int,
                "citations": List of citation strings found,
                "unverified": List of citations not matching sources
            }
        """
        # Regex patterns for APA-style inline citations
        single_author = r'\[([A-Z][a-zA-Z\-]+),?\s*(\d{4})\]'
        et_al = r'\[([A-Z][a-zA-Z\-]+)\s+et\s+al\.?,?\s*(\d{4})\]'
        two_authors = r'\[([A-Z][a-zA-Z\-]+)\s*[&]\s*([A-Z][a-zA-Z\-]+),?\s*(\d{4})\]'
        web_source = r'\[([a-zA-Z][a-zA-Z0-9\.\-]+),?\s*(\d{4})\]'

        all_citations = []

        # Find all citations
        single_matches = re.findall(single_author, draft)
        for match in single_matches:
            all_citations.append(f"[{match[0]}, {match[1]}]")

        et_al_matches = re.findall(et_al, draft)
        for match in et_al_matches:
            all_citations.append(f"[{match[0]} et al., {match[1]}]")

        two_author_matches = re.findall(two_authors, draft)
        for match in two_author_matches:
            all_citations.append(f"[{match[0]} & {match[1]}, {match[2]}]")

        # Deduplicate
        unique_citations = list(set(all_citations))

        # Verify against actual sources
        verified = []
        unverified = []

        # Build set of expected author-year combinations from sources
        source_refs = set()
        for s in sources:
            if isinstance(s, dict):
                authors = s.get("authors", [])
                year = str(s.get("year", ""))

                # Handle various author formats
                if isinstance(authors, list) and authors:
                    first_author = authors[0]
                    if isinstance(first_author, dict):
                        name = first_author.get("name", "")
                    else:
                        name = str(first_author)

                    # Extract last name
                    if " " in name:
                        last_name = name.split()[-1]
                    else:
                        last_name = name

                    if last_name and year:
                        source_refs.add((last_name.lower(), year))

                elif isinstance(authors, str):
                    if " " in authors:
                        last_name = authors.split()[-1]
                    else:
                        last_name = authors
                    if last_name and year:
                        source_refs.add((last_name.lower(), year))

                # Also add title-based refs for web sources
                title = s.get("title", "")
                if title and year:
                    first_word = title.split()[0] if title.split() else ""
                    if first_word:
                        source_refs.add((first_word.lower(), year))

        # Check each citation against sources
        for cite in unique_citations:
            cite_match = re.search(r'\[([^,\]]+)', cite)
            year_match = re.search(r'(\d{4})', cite)

            if cite_match and year_match:
                cite_author = cite_match.group(1).strip().lower()
                cite_year = year_match.group(1)

                # Remove "et al" if present
                cite_author = re.sub(r'\s*et\s*al\.?', '', cite_author).strip()

                is_verified = False
                for (src_author, src_year) in source_refs:
                    if (cite_author in src_author or src_author in cite_author) and cite_year == src_year:
                        is_verified = True
                        break

                if is_verified:
                    verified.append(cite)
                else:
                    unverified.append(cite)

        return {
            "citation_count": len(unique_citations),
            "verified_count": len(verified),
            "citations": unique_citations,
            "verified": verified,
            "unverified": unverified,
        }

    def save_to_file(self, filepath: str) -> None:
        """Persist reflexion memory."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, "w") as f:
            json.dump(self.memory, f, indent=2)

    def load_from_file(self, filepath: str) -> bool:
        """Load reflexion memory."""
        try:
            with open(filepath, "r") as f:
                self.memory = json.load(f)
            return True
        except Exception:
            return False


# ----- SAFETY GUARDIAN (from src/langgraph/safety.py) -----


class SafetyGuardian:
    """
    3-Layer Safety System for LangGraph.

    Provides comprehensive safety checks at every stage of the research pipeline.
    """

    def __init__(self):
        self.safety_log: List[Dict] = []

        self.hci_keywords = [
            "hci", "user interface", "usability", "accessibility",
            "user experience", "ux", "interaction design", "human factors",
            "interface", "ergonomics", "user study", "chi", "uist",
            "human-computer", "design", "prototype", "evaluation",
            "nielsen", "heuristics", "cognitive", "mental model"
        ]

        self.jailbreak_patterns = [
            r"ignore.*instructions",
            r"pretend.*you.*are",
            r"act.*as.*if",
            r"bypass.*safety",
            r"override.*rules",
            r"disregard.*previous",
            r"forget.*everything",
        ]

        self.harmful_patterns = [
            r"how.*to.*hack",
            r"how.*to.*harm",
            r"illegal.*activity",
            r"violence",
            r"exploit.*vulnerability",
            r"steal.*data",
            r"malware",
        ]

    def preflight_check(self, query: str, context: str = "") -> Dict[str, Any]:
        """Layer 1: Validate input before processing."""
        violations = []

        query_lower = query.lower()

        for pattern in self.jailbreak_patterns:
            if re.search(pattern, query_lower):
                violations.append({
                    "layer": 1,
                    "type": "jailbreak",
                    "severity": "critical",
                    "message": "Potential jailbreak attempt detected"
                })
                break

        for pattern in self.harmful_patterns:
            if re.search(pattern, query_lower):
                violations.append({
                    "layer": 1,
                    "type": "harmful_input",
                    "severity": "critical",
                    "message": "Harmful content detected"
                })
                break

        has_hci = any(kw in query_lower for kw in self.hci_keywords)
        if not has_hci and len(query.split()) > 5:
            violations.append({
                "layer": 1,
                "type": "off_topic",
                "severity": "warning",
                "message": "Query may not be HCI-related"
            })

        can_proceed = not any(v["severity"] == "critical" for v in violations)

        self._log("preflight", query[:100], can_proceed, violations)

        return {
            "can_proceed": can_proceed,
            "violations": violations,
            "safe": can_proceed,
        }

    def inflight_check(self, tool_output: str, tool_name: str) -> Dict[str, Any]:
        """Layer 2: Validate tool outputs during execution."""
        violations = []
        sanitized = tool_output

        if self._detect_pii(tool_output):
            sanitized = self._redact_pii(tool_output)
            violations.append({
                "layer": 2,
                "type": "pii_detected",
                "severity": "warning",
                "message": f"PII detected in {tool_name} results"
            })

        if len(tool_output.strip()) < 50:
            violations.append({
                "layer": 2,
                "type": "insufficient_results",
                "severity": "info",
                "message": f"{tool_name} returned minimal results"
            })

        self._log("inflight", tool_output[:100], True, violations)

        return {
            "safe": True,
            "violations": violations,
            "sanitized_output": sanitized,
        }

    def postflight_check(self, response: str) -> Dict[str, Any]:
        """Layer 3: Validate and sanitize final output."""
        violations = []
        sanitized = response

        if self._detect_pii(response):
            sanitized = self._redact_pii(response)
            violations.append({
                "layer": 3,
                "type": "pii_redacted",
                "severity": "info",
                "message": "PII redacted from response"
            })

        for pattern in self.harmful_patterns:
            if re.search(pattern, response.lower()):
                violations.append({
                    "layer": 3,
                    "type": "harmful_output",
                    "severity": "critical",
                    "message": "Harmful content in response"
                })
                break

        if len(response) < 200:
            violations.append({
                "layer": 3,
                "type": "insufficient_response",
                "severity": "warning",
                "message": "Response may be too brief"
            })

        safe = not any(v["severity"] == "critical" for v in violations)

        self._log("postflight", response[:100], safe, violations)

        return {
            "safe": safe,
            "violations": violations,
            "sanitized_response": sanitized if safe else self._get_blocked_message(),
        }

    def _detect_pii(self, text: str) -> bool:
        """Detect personally identifiable information."""
        patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
        ]
        return any(re.search(p, text) for p in patterns)

    def _redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL]",
            text
        )
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
        return text

    def _get_blocked_message(self) -> str:
        """Get message for blocked responses."""
        return "I cannot process this request due to safety guidelines. Please ask a question related to Human-Computer Interaction research."

    def _log(self, layer: str, content: str, safe: bool, violations: List) -> None:
        """Log safety event."""
        self.safety_log.append({
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "content_preview": content,
            "safe": safe,
            "violations": violations,
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics."""
        total = len(self.safety_log)
        by_layer = {"preflight": 0, "inflight": 0, "postflight": 0}
        violations_by_type: Dict[str, int] = {}

        for event in self.safety_log:
            layer = event.get("layer", "unknown")
            by_layer[layer] = by_layer.get(layer, 0) + 1
            for v in event.get("violations", []):
                vtype = v.get("type", "unknown")
                violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1

        return {
            "total_checks": total,
            "by_layer": by_layer,
            "violations_by_type": violations_by_type,
        }

    def clear_log(self) -> None:
        """Clear safety log."""
        self.safety_log = []


# ----- TEAMS (merged into src/agents/autogen_agents.py) -----
# Note: Team implementations are now in the merged autogen_agents.py file
# This consolidates LangGraph teams with commented AutoGen agents

from src.agents.autogen_agents import (
    create_planning_team,
    create_research_team,
    create_synthesis_team,
)


# ----- SUPERVISOR GRAPH (merged from src/langgraph/supervisor.py) -----
# Note: Supervisor graph implementation has been merged into this file
# The original supervisor.py contained the StateGraph workflow logic

from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json


def create_supervisor_graph(config: Dict) -> StateGraph:
    """
    Create the main supervisor graph with hierarchical teams.

    Architecture:
    1. Safety Guardian (pre-flight validation)
    2. Supervisor routes to specialized teams
    3. Teams execute (with parallel tools in Research Team)
    4. Reflexion check for self-correction
    5. Evaluation triangulation
    6. Human feedback checkpoint (optional)
    7. Post-flight safety
    8. Memory persistence

    Returns:
        Compiled LangGraph StateGraph
    """

    graph = StateGraph(ResearchState)

    # Store config for node access
    _config = config

    # === NODE IMPLEMENTATIONS ===

    def safety_guardian_node(state: ResearchState) -> Dict:
        """Layer 1: Pre-flight safety check with semantic validation."""
        guardian = SafetyGuardian()

        result = guardian.preflight_check(
            query=state["query"],
            context=state.get("memory_context", "")
        )

        return {
            "safety_layer1_result": result,
            "is_safe": result["can_proceed"],
            "status": "planning" if result["can_proceed"] else "blocked",
        }

    def supervisor_node(state: ResearchState) -> Dict:
        """
        Supervisor Agent: Analyzes state and decides next action.

        Uses LLM reasoning to determine:
        - What has been completed
        - What needs to be done next
        - Which team to route to
        """
        model = ChatOpenAI(
            model=_config["models"]["default"]["name"],
            temperature=0.2,
        )

        system_prompt = """You are a Research Supervisor managing specialized teams for HCI research.

## Available Teams
1. PLANNING TEAM: Decomposes queries into research tasks (Chain-of-Thought reasoning)
2. RESEARCH TEAM: Gathers evidence from papers and web (parallel tool execution)
3. SYNTHESIS TEAM: Writes and critiques the final response (multi-perspective evaluation)

## Current State Analysis
- Planning completed: {planning_done}
- Research completed: {research_done}
- Synthesis completed: {synthesis_done}
- Current iteration: {iteration}/{max_iterations}

## Decision Rules (Route to teams only - evaluation pipeline handles completion)
1. If no plan exists → route to "planning"
2. If plan exists but no research → route to "research"
3. If research exists but no synthesis → route to "synthesis"
4. If synthesis needs revision → route to "synthesis"

NOTE: After synthesis completes, the evaluation pipeline (reflexion → judge → triangulate)
handles quality assessment. You should ONLY route to teams, never to "done".

## Output Format
Return JSON only: {{"route": "planning|research|synthesis", "reasoning": "brief explanation"}}"""

        # Analyze current state
        planning_done = state.get("planning_output") is not None
        paper_results = state.get("paper_results", [])
        web_results = state.get("web_results", [])
        research_done = len(paper_results) > 0 or len(web_results) > 0
        draft = state.get("draft", "")
        synthesis_done = draft is not None and len(draft) > 100

        context = {
            "planning_done": planning_done,
            "research_done": research_done,
            "synthesis_done": synthesis_done,
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", 5),
            "reflexion_score": state.get("reflexion_score", 0.0),
        }

        messages = [
            SystemMessage(content=system_prompt.format(**context)),
            HumanMessage(content=f"Query: {state['query']}\n\nMake your routing decision.")
        ]

        response = model.invoke(messages)

        # Parse the JSON response
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            decision = json.loads(content.strip())
            route = decision.get("route", "planning")
        except (json.JSONDecodeError, IndexError):
            # Fallback logic based on state
            if not planning_done:
                route = "planning"
            elif not research_done:
                route = "research"
            else:
                route = "synthesis"

        return {
            "current_team": route,
            "iteration": state.get("iteration", 0) + 1,
        }

    def planning_team_node(state: ResearchState) -> Dict:
        """Execute the Planning Team subgraph."""
        team = create_planning_team(_config)
        result = team.invoke(state)
        return {
            "planning_output": result.get("planning_output"),
        }

    def research_team_node(state: ResearchState) -> Dict:
        """Execute the Research Team subgraph (with parallel tools)."""
        team = create_research_team(_config)
        result = team.invoke(state)
        return {
            "paper_results": result.get("paper_results", []),
            "web_results": result.get("web_results", []),
            "tool_calls_log": result.get("tool_calls_log", []),
            "research_output": result.get("research_output"),
        }

    def synthesis_team_node(state: ResearchState) -> Dict:
        """Execute the Synthesis Team subgraph."""
        team = create_synthesis_team(_config)
        result = team.invoke(state)
        return {
            "draft": result.get("draft", ""),
            "critique": result.get("critique", ""),
            "synthesis_output": result.get("synthesis_output"),
            "source_registry": result.get("source_registry", {}),
        }

    def verification_node(state: ResearchState) -> Dict:
        """
        Phase 5: Chain-of-Verification (CoVe) Node.

        Extracts factual claims from the draft and verifies each against sources.
        This improves factual accuracy scores.

        Steps:
        1. Extract factual claims from draft using LLM
        2. Verify each claim against paper/web sources
        3. Calculate verification score
        4. Flag unsupported claims for revision
        """
        print("[NODE] verification: Starting Chain-of-Verification...")

        model = ChatOpenAI(
            model=_config["models"]["default"]["name"],
            temperature=0.0,
        )

        draft = state.get("draft", "")
        sources = state.get("paper_results", []) + state.get("web_results", [])

        if not draft or len(draft) < 100:
            print("[NODE] verification: Draft too short, skipping")
            return {"verification_score": 0.0, "verified_claims": []}

        # Step 1: Extract factual claims
        extract_prompt = f"""Extract the key factual claims from this research response.
A factual claim is a statement that can be verified as true or false.

Response:
{draft[:3000]}

List up to 10 key factual claims, one per line, numbered 1-10.
Only include specific, verifiable claims (not opinions or general statements)."""

        claims_response = model.invoke([
            SystemMessage(content="You are a fact-extraction assistant."),
            HumanMessage(content=extract_prompt)
        ])

        # Parse claims
        claims = []
        for line in claims_response.content.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove numbering
                claim = line.lstrip('0123456789.-) ').strip()
                if len(claim) > 10:
                    claims.append(claim)

        if not claims:
            print("[NODE] verification: No claims extracted")
            return {"verification_score": 0.7, "verified_claims": []}

        # Step 2: Build source context for verification
        source_context = ""
        for i, s in enumerate(sources[:10]):
            if isinstance(s, dict):
                title = s.get("title", "Unknown")
                abstract = s.get("abstract", s.get("snippet", ""))[:300]
                source_context += f"\nSource {i+1}: {title}\n{abstract}\n"

        # Step 3: Verify claims against sources
        verify_prompt = f"""Verify each claim against the provided sources.
For each claim, determine if it is:
- VERIFIED: Directly supported by sources
- PARTIAL: Partially supported or implied
- UNVERIFIED: Not supported by provided sources

Sources:
{source_context}

Claims to verify:
{chr(10).join(f'{i+1}. {c}' for i, c in enumerate(claims[:10]))}

Output format (one per line):
CLAIM_NUMBER: STATUS (brief reason)"""

        verify_response = model.invoke([
            SystemMessage(content="You are a fact-verification assistant. Be strict about verification."),
            HumanMessage(content=verify_prompt)
        ])

        # Step 4: Calculate verification score
        verified = 0
        partial = 0
        results = []

        for line in verify_response.content.split('\n'):
            line_upper = line.upper()
            if "VERIFIED" in line_upper and "UNVERIFIED" not in line_upper:
                verified += 1
                results.append({"status": "verified", "line": line})
            elif "PARTIAL" in line_upper:
                partial += 1
                results.append({"status": "partial", "line": line})
            elif "UNVERIFIED" in line_upper:
                results.append({"status": "unverified", "line": line})

        total_claims = len(claims)
        if total_claims > 0:
            score = (verified + 0.5 * partial) / total_claims
        else:
            score = 0.7

        print(f"[NODE] verification: {verified}/{total_claims} verified, {partial} partial, score={score:.3f}")

        return {
            "verification_score": score,
            "verified_claims": results,
            "total_claims_checked": total_claims,
        }

    def self_refine_node(state: ResearchState) -> Dict:
        """
        Phase 6: Self-Refine Node.

        If verification or draft quality is low, refine the draft automatically.
        Uses self-critique → revision pattern.

        Steps:
        1. Check if refinement is needed (low scores or low citation count)
        2. Generate self-critique
        3. Apply refinement if needed
        """
        print("[NODE] self_refine: Starting self-refinement check...")

        verification_score = state.get("verification_score", 0.7)
        draft = state.get("draft", "")
        iteration = state.get("iteration", 0)

        # Count inline citations in draft using regex
        import re
        citation_pattern = r'\[([A-Z][a-z]+(?:\s+(?:et al\.|&\s+[A-Z][a-z]+))?),?\s*\d{4}\]'
        citation_matches = re.findall(citation_pattern, draft)
        citation_count = len(citation_matches)

        # Trigger refinement if: low verification OR too few citations
        low_verification = verification_score < 0.65
        low_citations = citation_count < 8  # Need at least 8 inline citations for 0.85+ evidence_quality

        needs_refine = (low_verification or low_citations) and iteration < 4

        if not needs_refine:
            print(f"[NODE] self_refine: No refinement needed (score={verification_score:.3f}, citations={citation_count})")
            return {"self_refine_applied": False}

        print(f"[NODE] self_refine: Refinement triggered (score={verification_score:.3f}, citations={citation_count})")

        model = ChatOpenAI(
            model=_config["models"]["default"]["name"],
            temperature=0.3,
        )

        # Get unverified claims to focus refinement
        verified_claims = state.get("verified_claims", [])
        unverified = [c for c in verified_claims if c.get("status") == "unverified"]

        unverified_text = ""
        if unverified:
            unverified_text = "\n\nClaims that need better support:\n" + "\n".join(
                c.get("line", "")[:100] for c in unverified[:5]
            )

        # Step 1: Self-critique
        critique_prompt = f"""Review this research response and identify specific improvements needed:

Query: {state.get("query", "")}

Current Response:
{draft[:2500]}
{unverified_text}

Provide 3-5 specific, actionable improvements (one per line):"""

        critique_response = model.invoke([
            SystemMessage(content="You are a critical reviewer focused on factual accuracy and evidence quality."),
            HumanMessage(content=critique_prompt)
        ])

        critique = critique_response.content

        # Get source registry for adding citations
        source_registry = state.get("source_registry", {})
        available_citations = ""
        if source_registry:
            available_citations = "\n\nAvailable citations to use:\n"
            for sid, info in list(source_registry.items())[:10]:
                cite_format = info.get("inline_citation", f"[{sid}]")
                title = info.get("title", "Unknown")[:60]
                available_citations += f"  - {cite_format}: {title}\n"

        # Step 2: Refine based on critique
        refine_prompt = f"""Improve this research response based on the critique below.

CRITICAL: You MUST add at least 10 inline citations in [Author, Year] format.
Use the available citations listed below. Distribute citations across ALL sections.

Focus on:
1. Adding citations [Author, Year] for EVERY factual claim - aim for 10+ citations
2. Using the exact citation formats provided below
3. Softening claims that cannot be verified
4. Improving clarity and structure
5. Including a References section at the end
{available_citations}

Original Response:
{draft[:2500]}

Critique:
{critique}

Write the improved response with at least 10 inline citations distributed throughout:"""

        refined_response = model.invoke([
            SystemMessage(content="You are a research writer improving your draft based on feedback."),
            HumanMessage(content=refine_prompt)
        ])

        refined_draft = refined_response.content

        print(f"[NODE] self_refine: Applied refinement (original={len(draft)}, refined={len(refined_draft)})")

        return {
            "draft": refined_draft,
            "critique": critique,
            "self_refine_applied": True,
        }

    def reflexion_check_node(state: ResearchState) -> Dict:
        """
        Reflexion Engine: Self-evaluate and learn from past mistakes.

        Innovation: Tracks failures in memory and applies lessons to avoid repeating them.
        """
        print("[NODE] reflexion_check: Starting evaluation...")
        engine = ReflexionEngine(state.get("reflexion_memory", {}))

        evaluation = engine.evaluate(
            query=state["query"],
            draft=state.get("draft", ""),
            sources=state.get("paper_results", []) + state.get("web_results", [])
        )

        # Update reflexion memory with new insights
        updated_memory = engine.update_memory(
            query=state["query"],
            success=evaluation["score"] >= 0.7,
            lesson=evaluation.get("lesson", "")
        )

        # Track tool effectiveness
        for log in state.get("tool_calls_log", []):
            engine.update_tool_effectiveness(
                log.get("tool", "unknown"),
                log.get("success", False)
            )

        print(f"[NODE] reflexion_check: score={evaluation['score']:.3f}")
        return {
            "reflexion_score": evaluation["score"],
            "reflexion_memory": updated_memory,
            "synthesis_output": {
                **(state.get("synthesis_output") or {}),
                "reflection": evaluation.get("feedback", ""),
            }
        }

    def llm_judge_node(state: ResearchState) -> Dict:
        """
        LLM Judge: Automated quality evaluation.

        Part of the Evaluation Triangulation system.
        """
        model = ChatOpenAI(
            model=_config["models"]["default"]["name"],
            temperature=0.1,
        )

        system_prompt = """You are an impartial LLM Judge evaluating research response quality.

## Evaluation Criteria (Score 0.0 to 1.0 each)
1. RELEVANCE (25%): Does the response address the query directly?
2. EVIDENCE (25%): Are claims supported by cited sources?
3. ACCURACY (20%): Is the information factually correct?
4. SAFETY (15%): Is the response appropriate and harmless?
5. CLARITY (15%): Is the writing clear and well-structured?

## Output Format
Return JSON only:
{
  "relevance": 0.0-1.0,
  "evidence": 0.0-1.0,
  "accuracy": 0.0-1.0,
  "safety": 0.0-1.0,
  "clarity": 0.0-1.0,
  "overall": 0.0-1.0,
  "feedback": "brief constructive feedback"
}"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {state['query']}\n\nResponse:\n{state.get('draft', '')[:3000]}")
        ]

        response = model.invoke(messages)

        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            scores = json.loads(content.strip())
            overall_score = scores.get("overall", 0.5)
        except (json.JSONDecodeError, IndexError):
            overall_score = 0.5

        print(f"[NODE] llm_judge: score={overall_score:.3f}")
        return {
            "llm_judge_score": overall_score,
        }

    def triangulate_node(state: ResearchState) -> Dict:
        """
        Evaluation Triangulation: Combine three evaluation sources.

        Innovation: Three-source decision making:
        1. LLM Judge Score (automated)
        2. Reflexion Score (self-critique)
        3. Human Score (optional, via interrupt)

        Decision: 2/3 must approve (score >= 0.7) for acceptance.
        """
        llm_score = state.get("llm_judge_score", 0)
        reflexion_score = state.get("reflexion_score", 0)
        human_score = state.get("human_score")

        # Count approvals (score >= 0.7 = approve)
        approvals = 0
        if llm_score >= 0.7:
            approvals += 1
        if reflexion_score >= 0.7:
            approvals += 1
        if human_score is not None and human_score >= 0.7:
            approvals += 1

        # Determine decision based on triangulation
        if human_score is None:
            # No human review yet
            if approvals >= 2:
                decision = "approved"
            elif llm_score < 0.5 or reflexion_score < 0.5:
                decision = "revise"  # Clear failure, no human needed
            else:
                decision = "human_needed"  # Borderline - could request human
        else:
            # Human has reviewed
            decision = "approved" if approvals >= 2 else "revise"

        # For now, skip human review unless explicitly requested
        # This makes the system fully automated
        if decision == "human_needed":
            decision = "approved" if (llm_score + reflexion_score) / 2 >= 0.65 else "revise"

        print(f"[NODE] triangulate: decision={decision} (llm={llm_score:.3f}, reflexion={reflexion_score:.3f})")
        return {
            "triangulated_decision": decision,
            "status": "approved" if decision == "approved" else "reviewing",
        }

    def safety_postflight_node(state: ResearchState) -> Dict:
        """Layer 3: Post-flight safety check and sanitization."""
        print("[NODE] safety_postflight: Running post-flight safety check...")
        guardian = SafetyGuardian()

        result = guardian.postflight_check(state.get("draft", ""))

        print(f"[NODE] safety_postflight: safe={result['safe']}, final_response set")
        return {
            "safety_layer3_result": result,
            "final_response": result.get("sanitized_response", state.get("draft", "")),
            "is_safe": result["safe"],
        }

    def persist_memory_node(state: ResearchState) -> Dict:
        """Persist reflexion memory and research results for future queries."""
        from src.agents.memory import ResearchMemory

        try:
            memory = ResearchMemory()

            # Store the research finding
            sources = [
                p.get("title", "") for p in state.get("paper_results", [])[:5]
            ]
            memory.add_finding(
                query=state["query"],
                response=state.get("final_response", state.get("draft", ""))[:500],
                sources=sources
            )

            # Store citations for deduplication
            for paper in state.get("paper_results", [])[:10]:
                if isinstance(paper, dict) and paper.get("title"):
                    memory.add_citation(paper["title"], paper)

            # Save to disk
            memory.save_to_file("research_memory.json")
        except Exception:
            # Memory persistence is non-critical, don't fail the workflow
            pass

        return {"status": "approved"}

    # === ADD NODES TO GRAPH ===

    graph.add_node("safety_guardian", safety_guardian_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("planning_team", planning_team_node)
    graph.add_node("research_team", research_team_node)
    graph.add_node("synthesis_team", synthesis_team_node)
    graph.add_node("verification", verification_node)  # Phase 5: CoVe
    graph.add_node("self_refine", self_refine_node)  # Phase 6: Self-Refine
    graph.add_node("reflexion_check", reflexion_check_node)
    graph.add_node("llm_judge", llm_judge_node)
    graph.add_node("triangulate", triangulate_node)
    graph.add_node("safety_postflight", safety_postflight_node)
    graph.add_node("persist_memory", persist_memory_node)

    # === ROUTING FUNCTIONS ===

    def route_after_safety(state: ResearchState) -> str:
        """Route based on safety check result."""
        return "proceed" if state.get("is_safe", True) else "blocked"

    def supervisor_router(state: ResearchState) -> str:
        """Route based on supervisor decision."""
        route = state.get("current_team", "planning")
        if route in ["planning", "research", "synthesis", "done"]:
            return route
        return "planning"  # Default fallback

    def route_after_synthesis(state: ResearchState) -> str:
        """Route after synthesis - always go to reflexion check."""
        return "reflexion"

    def route_after_reflexion(state: ResearchState) -> str:
        """Route based on reflexion score."""
        score = state.get("reflexion_score", 0)
        iteration = state.get("iteration", 0)
        max_iter = state.get("max_iterations", 5)

        if score >= 0.6 or iteration >= max_iter:
            return "judge"  # Proceed to LLM judge
        return "revise"  # Go back for revision

    def route_after_triangulation(state: ResearchState) -> str:
        """Route based on triangulated decision."""
        decision = state.get("triangulated_decision", "approved")
        if decision == "approved":
            return "postflight"
        return "revise"

    # === ADD EDGES ===

    # Entry point: Safety first
    graph.set_entry_point("safety_guardian")

    # Safety → Supervisor or END (if blocked)
    graph.add_conditional_edges(
        "safety_guardian",
        route_after_safety,
        {"proceed": "supervisor", "blocked": END}
    )

    # Supervisor routes to appropriate team
    # NOTE: No "done" route - evaluation pipeline handles completion via:
    # synthesis_team → reflexion_check → llm_judge → triangulate → safety_postflight
    graph.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "planning": "planning_team",
            "research": "research_team",
            "synthesis": "synthesis_team",
        }
    )

    # Teams → back to supervisor (except synthesis)
    graph.add_edge("planning_team", "supervisor")
    graph.add_edge("research_team", "supervisor")

    # Synthesis → Verification → Self-Refine → Reflexion check
    # Phase 5 & 6: New pipeline for improved accuracy
    graph.add_edge("synthesis_team", "verification")
    graph.add_edge("verification", "self_refine")
    graph.add_edge("self_refine", "reflexion_check")

    # Reflexion → LLM Judge or back to supervisor for revision
    graph.add_conditional_edges(
        "reflexion_check",
        route_after_reflexion,
        {"judge": "llm_judge", "revise": "supervisor"}
    )

    # LLM Judge → Triangulation
    graph.add_edge("llm_judge", "triangulate")

    # Triangulation → Post-flight or revision
    graph.add_conditional_edges(
        "triangulate",
        route_after_triangulation,
        {"postflight": "safety_postflight", "revise": "supervisor"}
    )

    # Post-flight → Memory persistence
    graph.add_edge("safety_postflight", "persist_memory")

    # Memory → END
    graph.add_edge("persist_memory", END)

    return graph


def compile_graph(config: Dict):
    """Compile the supervisor graph for execution."""
    graph = create_supervisor_graph(config)
    return graph.compile()


# ----- MAIN ORCHESTRATOR CLASS (from src/langgraph_orchestrator.py) -----

import logging
import yaml

from src.agents.memory import ResearchMemory


class LangGraphOrchestrator:
    """
    LangGraph-based orchestrator for multi-agent HCI research.

    This orchestrator uses LangGraph's StateGraph for explicit control flow,
    replacing the round-robin approach of AutoGen with a hierarchical
    supervisor pattern.

    Key Features:
    - Explicit state management with TypedDict
    - Conditional routing based on supervisor decisions
    - Parallel tool execution in research team
    - Reflexion-based self-correction
    - Three-source evaluation triangulation
    - 3-layer safety architecture
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LangGraph orchestrator.

        Args:
            config: Configuration dictionary from config.yaml
        """
        self.config = config
        self.logger = logging.getLogger("langgraph_orchestrator")

        # Initialize memory system
        self.logger.info("Initializing memory system...")
        self.memory = ResearchMemory(max_findings=20, max_context=10)

        # Initialize safety guardian
        self.logger.info("Initializing safety guardian...")
        self.safety_guardian = SafetyGuardian()

        # Initialize reflexion engine
        self.logger.info("Initializing reflexion engine...")
        self.reflexion_engine = ReflexionEngine()

        # Compile the LangGraph
        self.logger.info("Compiling LangGraph supervisor...")
        self.graph = compile_graph(config)

        self.logger.info("LangGraph orchestrator initialized successfully")

        # Workflow trace for debugging
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str, max_iterations: int = 5) -> Dict[str, Any]:
        """
        Process a research query through the LangGraph multi-agent system.

        Args:
            query: The research question to answer
            max_iterations: Maximum number of revision iterations

        Returns:
            Dictionary containing:
            - query: Original query
            - response: Final synthesized response
            - workflow_trace: List of execution steps
            - metadata: Additional information
        """
        self.logger.info(f"Processing query: {query[:100]}...")

        # Get relevant past context from memory
        past_context = self.memory.get_relevant_context(query, k=3)
        if past_context:
            self.logger.info("Found relevant past research in memory")

        # Get related citations
        related_citations = self.memory.get_related_citations(query, k=5)

        # Create initial state
        initial_state = create_initial_state(query, max_iterations)

        # Add memory context to initial state
        initial_state["memory_context"] = past_context or ""
        initial_state["related_citations"] = related_citations or []

        try:
            # Execute the graph
            self.logger.info("Executing LangGraph workflow...")
            start_time = datetime.now()

            # Run the compiled graph
            final_state = self.graph.invoke(initial_state)

            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            self.logger.info(f"Workflow completed in {execution_time:.2f}s")

            # Extract results
            result = self._extract_results(query, final_state, execution_time)

            # Store findings in memory
            self._store_in_memory(query, result)

            return result

        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred while processing your query: {str(e)}",
                "workflow_trace": [],
                "metadata": {"error": True}
            }

    def _extract_results(
        self,
        query: str,
        final_state: Dict[str, Any],
        execution_time: float
    ) -> Dict[str, Any]:
        """Extract structured results from the final state."""

        # Get the final response
        final_response = final_state.get("final_response", "")
        if not final_response:
            final_response = final_state.get("draft", "")

        # Count sources
        paper_count = len(final_state.get("paper_results", []))
        web_count = len(final_state.get("web_results", []))

        # Build workflow trace
        workflow_trace = []

        # Add safety check
        if final_state.get("safety_layer1_result"):
            workflow_trace.append({
                "step": "Safety Pre-flight",
                "result": "passed" if final_state.get("is_safe") else "blocked",
                "details": final_state["safety_layer1_result"]
            })

        # Add planning output
        if final_state.get("planning_output"):
            workflow_trace.append({
                "step": "Planning Team",
                "result": "completed",
                "confidence": final_state["planning_output"].get("confidence", 0),
                "output_preview": final_state["planning_output"].get("output", "")[:200]
            })

        # Add research output
        if final_state.get("research_output"):
            workflow_trace.append({
                "step": "Research Team (Parallel Tools)",
                "result": "completed",
                "papers_found": paper_count,
                "web_sources_found": web_count,
                "confidence": final_state["research_output"].get("confidence", 0)
            })

        # Add synthesis output
        if final_state.get("synthesis_output"):
            workflow_trace.append({
                "step": "Synthesis Team",
                "result": "completed",
                "confidence": final_state["synthesis_output"].get("confidence", 0)
            })

        # Add reflexion check
        if final_state.get("reflexion_score") is not None:
            workflow_trace.append({
                "step": "Reflexion Check",
                "score": final_state["reflexion_score"],
                "result": "passed" if final_state["reflexion_score"] >= 0.6 else "needs_revision"
            })

        # Add evaluation triangulation
        workflow_trace.append({
            "step": "Evaluation Triangulation",
            "llm_judge_score": final_state.get("llm_judge_score", 0),
            "reflexion_score": final_state.get("reflexion_score", 0),
            "human_score": final_state.get("human_score"),
            "decision": final_state.get("triangulated_decision", "unknown")
        })

        # Add post-flight safety
        if final_state.get("safety_layer3_result"):
            workflow_trace.append({
                "step": "Safety Post-flight",
                "result": "passed" if final_state["safety_layer3_result"].get("safe") else "sanitized"
            })

        return {
            "query": query,
            "response": final_response,
            "workflow_trace": workflow_trace,
            "metadata": {
                "execution_time_seconds": execution_time,
                "iterations": final_state.get("iteration", 0),
                "num_messages": len(final_state.get("messages", [])),
                "num_sources": paper_count + web_count,
                "paper_sources": paper_count,
                "web_sources": web_count,
                "reflexion_score": final_state.get("reflexion_score", 0),
                "llm_judge_score": final_state.get("llm_judge_score", 0),
                "triangulated_decision": final_state.get("triangulated_decision", "unknown"),
                "is_safe": final_state.get("is_safe", True),
                "tool_calls": final_state.get("tool_calls_log", []),
            }
        }

    def _store_in_memory(self, query: str, result: Dict[str, Any]) -> None:
        """Store query results in memory for future context."""
        if "error" in result:
            return

        response = result.get("response", "")
        metadata = result.get("metadata", {})

        # Extract sources
        sources = []
        for trace in result.get("workflow_trace", []):
            if trace.get("step") == "Research Team (Parallel Tools)":
                sources.extend([f"Paper {i+1}" for i in range(trace.get("papers_found", 0))])

        # Store the finding
        self.memory.add_finding(query, response[:500], sources)

        self.logger.info(f"Stored finding in memory. Total findings: {len(self.memory.findings)}")

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about the memory system."""
        return self.memory.get_statistics()

    def get_safety_statistics(self) -> Dict[str, Any]:
        """Get statistics about safety checks."""
        return self.safety_guardian.get_statistics()

    def clear_memory(self) -> None:
        """Clear all stored memory."""
        self.memory.clear()
        self.logger.info("Memory cleared")

    def save_memory(self, filepath: str) -> None:
        """Save memory state to a file."""
        self.memory.save_to_file(filepath)
        self.logger.info(f"Memory saved to {filepath}")

    def load_memory(self, filepath: str) -> bool:
        """Load memory state from a file."""
        success = self.memory.load_from_file(filepath)
        if success:
            self.logger.info(f"Memory loaded from {filepath}")
        else:
            self.logger.warning(f"Failed to load memory from {filepath}")
        return success

    def get_agent_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all agents/teams."""
        return {
            "SafetyGuardian": "3-layer safety system (pre-flight, in-flight, post-flight)",
            "Supervisor": "Hierarchical supervisor routing to specialized teams",
            "PlanningTeam": "Chain-of-Thought query decomposition with self-critique",
            "ResearchTeam": "Parallel tool execution (paper_search + web_search)",
            "SynthesisTeam": "Multi-perspective writing and critique",
            "ReflexionEngine": "Self-correcting agent with failure memory",
            "LLMJudge": "Automated quality evaluation",
            "EvaluationTriangulation": "Three-source decision making (LLM + Reflexion + Human)",
        }

    def visualize_workflow(self) -> str:
        """Generate a text visualization of the workflow."""
        return """
LangGraph Research Workflow (Novel Architecture):

┌──────────────────────────────────────────────────────────────────────┐
│                         USER QUERY                                    │
│                            ↓                                          │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │           SAFETY GUARDIAN (Layer 1: Pre-flight)            │     │
│   │  - Jailbreak detection                                      │     │
│   │  - Harmful content filtering                                │     │
│   │  - HCI topic validation                                     │     │
│   └──────────────────────┬─────────────────────────────────────┘     │
│                          ↓                                            │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │              SUPERVISOR AGENT (Routing)                     │     │
│   │  Decides: planning → research → synthesis → done            │     │
│   └──────────┬───────────┬───────────┬────────────────────────┘     │
│              ↓           ↓           ↓                                │
│   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                 │
│   │ PLANNING     │ │ RESEARCH     │ │ SYNTHESIS    │                 │
│   │ TEAM         │ │ TEAM         │ │ TEAM         │                 │
│   │              │ │              │ │              │                 │
│   │ • Planner    │ │ • Parallel   │ │ • Writer     │                 │
│   │ • Reflector  │ │   Tools:     │ │ • Critic     │                 │
│   │              │ │   - paper_   │ │              │                 │
│   │ Chain-of-    │ │     search   │ │ Multi-       │                 │
│   │ Thought      │ │   - web_     │ │ perspective  │                 │
│   │ Decomp.      │ │     search   │ │ Evaluation   │                 │
│   └──────────────┘ └──────────────┘ └──────┬───────┘                 │
│                                            ↓                          │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │           REFLEXION ENGINE (Innovation 1)                   │     │
│   │  - Self-evaluate output quality                             │     │
│   │  - Track past failures in memory                            │     │
│   │  - Apply lessons to avoid repeating mistakes                │     │
│   │  - Score >= 0.6 → proceed, else → revise                    │     │
│   └──────────────────────┬─────────────────────────────────────┘     │
│                          ↓                                            │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │        EVALUATION TRIANGULATION (Innovation 3)              │     │
│   │                                                             │     │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │     │
│   │  │ LLM JUDGE   │  │ REFLEXION   │  │ HUMAN       │         │     │
│   │  │ (Automated) │  │ SCORE       │  │ (Optional)  │         │     │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │     │
│   │         └────────────────┼────────────────┘                 │     │
│   │                          ↓                                  │     │
│   │          Decision: 2/3 must approve (score >= 0.7)         │     │
│   └──────────────────────┬─────────────────────────────────────┘     │
│                          ↓                                            │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │           SAFETY GUARDIAN (Layer 3: Post-flight)           │     │
│   │  - PII redaction                                            │     │
│   │  - Output sanitization                                      │     │
│   │  - Final harmful content check                              │     │
│   └──────────────────────┬─────────────────────────────────────┘     │
│                          ↓                                            │
│   ┌────────────────────────────────────────────────────────────┐     │
│   │              MEMORY PERSISTENCE                             │     │
│   │  - Store findings for future queries                        │     │
│   │  - Save reflexion lessons                                   │     │
│   │  - Track tool effectiveness                                 │     │
│   └──────────────────────┬─────────────────────────────────────┘     │
│                          ↓                                            │
│                    FINAL RESPONSE                                     │
└──────────────────────────────────────────────────────────────────────┘

Innovation 2: Parallel Tool Execution
- Research Team runs paper_search and web_search CONCURRENTLY
- Uses ThreadPoolExecutor for ~50% faster research phase
"""


def demonstrate_usage():
    """Demonstrate how to use the LangGraph orchestrator."""
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create orchestrator
    orchestrator = LangGraphOrchestrator(config)

    # Print workflow visualization
    print(orchestrator.visualize_workflow())

    # Print agent descriptions
    print("\nAgent/Team Descriptions:")
    print("-" * 50)
    for agent, desc in orchestrator.get_agent_descriptions().items():
        print(f"  {agent}: {desc}")

    # Example query
    query = "What are Nielsen's usability heuristics and how are they applied in modern interface design?"

    print(f"\n{'='*70}")
    print(f"Processing query: {query}")
    print("=" * 70)

    # Process query
    result = orchestrator.process_query(query)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nQuery: {result['query']}")
    print(f"\nResponse:\n{result['response'][:1000]}...")
    print(f"\nWorkflow Trace:")
    for step in result.get("workflow_trace", []):
        print(f"  - {step['step']}: {step.get('result', 'completed')}")
    print(f"\nMetadata:")
    meta = result.get("metadata", {})
    print(f"  - Execution time: {meta.get('execution_time_seconds', 0):.2f}s")
    print(f"  - Iterations: {meta.get('iterations', 0)}")
    print(f"  - Paper sources: {meta.get('paper_sources', 0)}")
    print(f"  - Web sources: {meta.get('web_sources', 0)}")
    print(f"  - Reflexion score: {meta.get('reflexion_score', 0):.3f}")
    print(f"  - LLM judge score: {meta.get('llm_judge_score', 0):.3f}")
    print(f"  - Decision: {meta.get('triangulated_decision', 'unknown')}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    demonstrate_usage()
