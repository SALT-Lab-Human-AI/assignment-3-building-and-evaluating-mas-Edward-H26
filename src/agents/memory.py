"""
Multi-Agent Memory System
Provides shared memory capabilities for the research agent system.

Implements:
- Research findings storage and retrieval
- Citation deduplication
- Context window management
- Relevant context extraction for new queries
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os


class ResearchMemory:
    """
    Shared memory store for multi-agent research system.

    Stores:
    - Research findings from previous queries
    - Citation database for deduplication
    - Rolling context window
    - User preferences (future extension)

    Memory is designed to persist across queries within a session
    and can optionally be saved to disk for cross-session persistence.
    """

    def __init__(self, max_findings: int = 20, max_context: int = 10):
        """
        Initialize the research memory.

        Args:
            max_findings: Maximum number of research findings to store
            max_context: Maximum number of context items in rolling window
        """
        self.findings: List[Dict[str, Any]] = []
        self.citations: Dict[str, Dict[str, Any]] = {}
        self.context_window: List[str] = []
        self.max_findings = max_findings
        self.max_context = max_context
        self.session_start = datetime.now().isoformat()

    def add_finding(self, query: str, response: str, sources: List[str]) -> None:
        """
        Store a research finding from a completed query.

        Args:
            query: The original research query
            response: The synthesized response
            sources: List of source URLs/references used
        """
        finding = {
            "query": query,
            "response": response[:500],  # Truncate for memory efficiency
            "sources": sources[:10],  # Limit sources
            "timestamp": datetime.now().isoformat(),
            "query_keywords": list(self._extract_keywords(query))  # Convert set to list for JSON
        }

        self.findings.append(finding)

        # Maintain maximum findings limit (FIFO)
        if len(self.findings) > self.max_findings:
            self.findings = self.findings[-self.max_findings:]

    def add_citation(self, title: str, metadata: Dict[str, Any]) -> bool:
        """
        Store a citation for deduplication tracking.

        Args:
            title: Citation title (used as key)
            metadata: Citation metadata (authors, year, url, etc.)

        Returns:
            True if new citation, False if duplicate
        """
        # Normalize title for deduplication
        normalized_title = title.lower().strip()

        if normalized_title in self.citations:
            # Update access count for existing citation
            self.citations[normalized_title]["access_count"] = \
                self.citations[normalized_title].get("access_count", 1) + 1
            return False

        self.citations[normalized_title] = {
            **metadata,
            "added_at": datetime.now().isoformat(),
            "access_count": 1
        }
        return True

    def get_citation(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a citation by title.

        Args:
            title: Citation title to look up

        Returns:
            Citation metadata if found, None otherwise
        """
        normalized_title = title.lower().strip()
        return self.citations.get(normalized_title)

    def get_all_citations(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored citations."""
        return self.citations.copy()

    def get_related_citations(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Get citations related to query keywords.

        Uses keyword matching to find citations whose titles
        overlap with the query terms.

        Args:
            query: Query to find related citations for
            k: Maximum number of citations to return

        Returns:
            List of related citation dictionaries with title and metadata
        """
        if not self.citations:
            return []

        query_words = self._extract_keywords(query)
        scored_citations = []

        for title, meta in self.citations.items():
            # Score by keyword overlap with title
            title_words = self._extract_keywords(title)
            overlap = len(query_words & title_words)

            if overlap > 0:
                citation_entry = {
                    "title": title,
                    "year": meta.get("year"),
                    "authors": meta.get("authors", []),
                    "url": meta.get("url", ""),
                    "relevance_score": overlap
                }
                scored_citations.append((overlap, citation_entry))

        # Sort by relevance score (descending)
        scored_citations.sort(key=lambda x: x[0], reverse=True)

        return [entry for _, entry in scored_citations[:k]]

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant past findings for context augmentation.

        Uses keyword overlap scoring to find the most relevant
        previous research findings.

        Args:
            query: Current query to find context for
            k: Number of relevant findings to return

        Returns:
            Formatted string of relevant past findings
        """
        if not self.findings:
            return ""

        query_keywords = self._extract_keywords(query)

        # Score findings by keyword overlap
        scored_findings = []
        for finding in self.findings:
            # Convert stored list back to set for comparison
            past_keywords = set(finding.get("query_keywords", []))
            overlap = len(query_keywords & past_keywords)

            # Also check for direct word matches in response
            response_words = set(finding.get("response", "").lower().split())
            response_overlap = len(query_keywords & response_words)

            total_score = overlap * 2 + response_overlap  # Weight query match higher

            if total_score > 0:
                scored_findings.append((total_score, finding))

        # Sort by relevance score (descending)
        scored_findings.sort(key=lambda x: x[0], reverse=True)

        # Format top-k findings as context
        context_parts = []
        for score, finding in scored_findings[:k]:
            query_preview = finding["query"][:50]
            if len(finding["query"]) > 50:
                query_preview += "..."

            response_preview = finding["response"][:200]
            if len(finding["response"]) > 200:
                response_preview += "..."

            context_parts.append(
                f"Previous research on '{query_preview}':\n{response_preview}"
            )

        return "\n\n".join(context_parts) if context_parts else ""

    def update_context_window(self, message: str) -> None:
        """
        Update the rolling context window with a new message.

        Args:
            message: Message to add to context window
        """
        # Truncate long messages
        truncated = message[:500] if len(message) > 500 else message
        self.context_window.append(truncated)

        # Maintain maximum context window size
        if len(self.context_window) > self.max_context:
            self.context_window = self.context_window[-self.max_context:]

    def get_context_window(self) -> List[str]:
        """Get the current context window."""
        return self.context_window.copy()

    def clear_context_window(self) -> None:
        """Clear the rolling context window."""
        self.context_window = []

    def _extract_keywords(self, text: str) -> set:
        """
        Extract keywords from text for relevance matching.

        Args:
            text: Text to extract keywords from

        Returns:
            Set of lowercase keywords
        """
        # Common stop words to filter out
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'shall', 'can', 'need', 'dare', 'ought', 'used', 'what', 'which',
            'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'it',
            'its', 'how', 'why', 'when', 'where', 'there', 'here', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'any', 'research', 'study', 'studies'
        }

        # Tokenize and filter
        words = text.lower().split()
        keywords = {
            word.strip('.,!?;:"\'()[]{}')
            for word in words
            if len(word) > 2 and word.lower() not in stop_words
        }

        return keywords

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "session_start": self.session_start,
            "num_findings": len(self.findings),
            "num_citations": len(self.citations),
            "context_window_size": len(self.context_window),
            "max_findings": self.max_findings,
            "max_context": self.max_context
        }

    def save_to_file(self, filepath: str) -> None:
        """
        Save memory state to a JSON file.

        Args:
            filepath: Path to save the memory state
        """
        state = {
            "session_start": self.session_start,
            "findings": self.findings,
            "citations": self.citations,
            "context_window": self.context_window,
            "saved_at": datetime.now().isoformat()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_from_file(self, filepath: str) -> bool:
        """
        Load memory state from a JSON file.

        Args:
            filepath: Path to load the memory state from

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            self.findings = state.get("findings", [])
            self.citations = state.get("citations", {})
            self.context_window = state.get("context_window", [])

            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def clear(self) -> None:
        """Clear all memory."""
        self.findings = []
        self.citations = {}
        self.context_window = []
        self.session_start = datetime.now().isoformat()


class AgentMemory:
    """
    Per-agent memory for tracking individual agent contributions.

    Allows tracking of:
    - Agent-specific findings
    - Tool usage patterns
    - Performance metrics
    """

    def __init__(self, agent_name: str):
        """
        Initialize agent-specific memory.

        Args:
            agent_name: Name of the agent this memory belongs to
        """
        self.agent_name = agent_name
        self.contributions: List[Dict[str, Any]] = []
        self.tool_usage: Dict[str, int] = {}
        self.created_at = datetime.now().isoformat()

    def add_contribution(self, content: str, contribution_type: str = "message") -> None:
        """
        Record an agent contribution.

        Args:
            content: The contribution content
            contribution_type: Type of contribution (message, tool_call, etc.)
        """
        self.contributions.append({
            "content": content[:500],
            "type": contribution_type,
            "timestamp": datetime.now().isoformat()
        })

    def record_tool_usage(self, tool_name: str) -> None:
        """
        Record a tool usage event.

        Args:
            tool_name: Name of the tool used
        """
        self.tool_usage[tool_name] = self.tool_usage.get(tool_name, 0) + 1

    def get_contribution_count(self) -> int:
        """Get the total number of contributions."""
        return len(self.contributions)

    def get_tool_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        return self.tool_usage.copy()

    def get_recent_contributions(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most recent contributions.

        Args:
            n: Number of recent contributions to return

        Returns:
            List of recent contributions
        """
        return self.contributions[-n:] if self.contributions else []
