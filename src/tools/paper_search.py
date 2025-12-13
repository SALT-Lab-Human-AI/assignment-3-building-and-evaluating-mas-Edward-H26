"""
Paper Search Tool
Integrates with Semantic Scholar API for academic paper search.

This tool provides academic paper search functionality using the
Semantic Scholar API, which offers free access to a large corpus
of academic papers.
"""

from typing import List, Dict, Any, Optional
import os
import time
import logging
import asyncio
import requests
from datetime import datetime


# HCI-relevant venues for filtering search results
HCI_VENUES = [
    "CHI", "UIST", "CSCW", "DIS", "IUI", "MobileHCI",
    "UbiComp", "INTERACT", "NordiCHI", "OzCHI",
    "Human-Computer Interaction", "ACM Computing Surveys",
    "International Journal of Human-Computer Studies",
    "Proceedings of the ACM on Human-Computer Interaction",
    "ACM Transactions on Computer-Human Interaction",
    "TOCHI", "IEEE VR", "ISMAR", "TEI", "IDC"
]

# HCI-relevant keywords for title/abstract filtering
HCI_KEYWORDS = [
    "hci", "human-computer interaction", "user interface", "usability",
    "user experience", "ux", "accessibility", "interaction design",
    "user study", "user research", "interface design", "human factors"
]


class PaperSearchTool:
    """
    Tool for searching academic papers via Semantic Scholar API.
    
    Semantic Scholar provides free access to academic papers with
    rich metadata including citations, abstracts, and author information.
    API key is optional but recommended for higher rate limits.
    """

    def __init__(self, max_results: int = 10):
        """
        Initialize paper search tool.

        Args:
            max_results: Maximum number of papers to return
        """
        self.max_results = max_results
        self.logger = logging.getLogger("tools.paper_search")

        # API key is optional for Semantic Scholar
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
        # Respect documented rate limit (default 1 req/sec unless key allows more)
        self.min_request_interval = float(os.getenv("SEMANTIC_SCHOLAR_REQUEST_INTERVAL", "1.0"))
        self.page_size = min(max_results, 20) if max_results > 0 else 10
        self._last_request_time = 0.0
        self._session = requests.Session()
        
        if not self.api_key:
            self.logger.info("No Semantic Scholar API key found. Using anonymous access (lower rate limits)")

    async def search(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers.

        Args:
            query: Search query
            year_from: Filter papers from this year onwards
            year_to: Filter papers up to this year
            min_citations: Minimum citation count
            **kwargs: Additional search parameters
                - fields: List of fields to retrieve

        Returns:
            List of papers with metadata format:
            {
                "paper_id": str,
                "title": str,
                "authors": List[{"name": str}],
                "year": int,
                "abstract": str,
                "citation_count": int,
                "url": str,
                "venue": str,
                "pdf_url": Optional[str],
            }
        """
        self.logger.info(f"Searching papers: {query}")

        fields = kwargs.get("fields", [
            "paperId", "title", "authors", "year", "abstract",
            "citationCount", "url", "venue", "openAccessPdf"
        ])

        papers: List[Dict[str, Any]] = []
        seen_ids = set()
        offset = 0

        while len(papers) < self.max_results:
            remaining = self.max_results - len(papers)
            page_limit = min(self.page_size, remaining)
            page = self._fetch_page(query, fields, offset, page_limit)

            if page is None:
                # Stop on hard errors
                break
            if not page:
                # No more results
                break

            parsed = self._parse_results(page, year_from, year_to, min_citations, query)

            for p in parsed:
                pid = p.get("paper_id")
                if pid and pid in seen_ids:
                    continue
                seen_ids.add(pid)
                papers.append(p)
                if len(papers) >= self.max_results:
                    break

            offset += page_limit

        self.logger.info(f"Found {len(papers)} papers")
        return papers

    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific paper.

        Args:
            paper_id: Semantic Scholar paper ID

        Returns:
            Detailed paper information
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            
            return {
                "paper_id": paper.paperId,
                "title": paper.title,
                "authors": [{"name": a.name} for a in paper.authors] if paper.authors else [],
                "year": paper.year,
                "abstract": paper.abstract,
                "citation_count": paper.citationCount,
                "url": paper.url,
                "venue": paper.venue,
                "pdf_url": paper.openAccessPdf.get("url") if paper.openAccessPdf else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting paper details: {e}")
            return {}

    async def get_citations(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers that cite this paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of citations to retrieve

        Returns:
            List of citing papers
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            citations = paper.citations[:limit] if paper.citations else []
            
            return [
                {
                    "paper_id": c.paperId,
                    "title": c.title,
                    "year": c.year,
                }
                for c in citations
            ]
        except Exception as e:
            self.logger.error(f"Error getting citations: {e}")
            return []

    async def get_references(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get papers referenced by this paper.

        Args:
            paper_id: Semantic Scholar paper ID
            limit: Maximum number of references to retrieve

        Returns:
            List of referenced papers
        """
        try:
            from semanticscholar import SemanticScholar
            
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            references = paper.references[:limit] if paper.references else []
            
            return [
                {
                    "paper_id": r.paperId,
                    "title": r.title,
                    "year": r.year,
                }
                for r in references
            ]
        except Exception as e:
            self.logger.error(f"Error getting references: {e}")
            return []

    def _parse_results(
        self,
        results: Any,
        year_from: Optional[int],
        year_to: Optional[int],
        min_citations: int,
        query: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Parse, filter, and rank search results from Semantic Scholar.

        Args:
            results: Raw results from Semantic Scholar API
            year_from: Minimum year filter
            year_to: Maximum year filter
            min_citations: Minimum citation count filter
            query: Original search query (for quality scoring)

        Returns:
            Filtered, scored, and sorted list of papers
        """
        papers = []

        for paper in results:
            # Skip papers without basic metadata
            if not paper:
                continue

            get_attr = paper.get if isinstance(paper, dict) else lambda key, default=None: getattr(paper, key, default)

            paper_dict = {
                "paper_id": get_attr("paperId"),
                "title": get_attr("title", "Unknown"),
                "authors": [{"name": a.get("name")} for a in get_attr("authors", []) if isinstance(a, dict)],
                "year": get_attr("year"),
                "abstract": get_attr("abstract", ""),
                "citation_count": get_attr("citationCount", 0),
                "url": get_attr("url", ""),
                "venue": get_attr("venue", ""),
                "pdf_url": (get_attr("openAccessPdf") or {}).get("url") if isinstance(get_attr("openAccessPdf"), dict) else None,
            }

            # Add HCI relevance flag and quality score
            paper_dict["is_hci_relevant"] = self._is_hci_relevant(paper_dict)
            if query:
                paper_dict["quality_score"] = self._calculate_quality_score(paper_dict, query)
            else:
                paper_dict["quality_score"] = 0.0

            papers.append(paper_dict)

        # Apply filters
        papers = self._filter_by_year(papers, year_from, year_to)
        papers = self._filter_by_citations(papers, min_citations)

        # Sort by quality score (highest first)
        papers.sort(key=lambda x: x.get("quality_score", 0), reverse=True)

        return papers

    def _filter_by_year(
        self,
        papers: List[Dict[str, Any]],
        year_from: Optional[int],
        year_to: Optional[int]
    ) -> List[Dict[str, Any]]:
        """Filter papers by publication year."""
        filtered = papers
        if year_from:
            filtered = [p for p in filtered if p.get("year") and p.get("year") >= year_from]
        if year_to:
            filtered = [p for p in filtered if p.get("year") and p.get("year") <= year_to]
        return filtered

    def _filter_by_citations(
        self,
        papers: List[Dict[str, Any]],
        min_citations: int
    ) -> List[Dict[str, Any]]:
        """Filter papers by citation count."""
        return [p for p in papers if p.get("citation_count", 0) >= min_citations]

    def _is_hci_relevant(self, paper: Dict[str, Any]) -> bool:
        """
        Check if paper is HCI-relevant by venue or keywords.

        Args:
            paper: Paper dictionary with venue, title, abstract fields

        Returns:
            True if paper appears HCI-relevant
        """
        venue = (paper.get("venue") or "").lower()
        title = (paper.get("title") or "").lower()
        abstract = (paper.get("abstract") or "").lower()

        # Check venue matches
        for hci_venue in HCI_VENUES:
            if hci_venue.lower() in venue:
                return True

        # Check HCI keywords in title or abstract
        text = f"{title} {abstract}"
        for keyword in HCI_KEYWORDS:
            if keyword in text:
                return True

        return False

    def _calculate_quality_score(self, paper: Dict[str, Any], query: str) -> float:
        """
        Calculate relevance and quality score for a paper.

        Args:
            paper: Paper dictionary
            query: Original search query

        Returns:
            Score between 0.0 and 1.0
        """
        score = 0.0

        # Citation score (0-0.3) - more citations = higher quality
        citations = paper.get("citation_count", 0) or 0
        score += min(0.3, citations / 100)

        # Recency score (0-0.2) - newer papers get higher scores
        year = paper.get("year") or 2000
        current_year = datetime.now().year
        years_old = current_year - year
        score += max(0, 0.2 - (years_old * 0.02))

        # Title relevance (0-0.3) - query word overlap with title
        query_words = set(query.lower().split())
        title_words = set((paper.get("title") or "").lower().split())
        overlap = len(query_words & title_words)
        score += min(0.3, overlap * 0.1)

        # HCI venue bonus (0-0.2)
        if self._is_hci_relevant(paper):
            score += 0.2

        return min(1.0, score)

    def _respect_rate_limit(self):
        """Sleep to enforce the minimum interval between API calls."""
        elapsed = time.time() - self._last_request_time
        wait_for = self.min_request_interval - elapsed
        if wait_for > 0:
            time.sleep(wait_for)

    def _fetch_page(
        self,
        query: str,
        fields: List[str],
        offset: int,
        limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch a single page from Semantic Scholar with backoff on 429."""
        params = {
            "query": query,
            "fields": ",".join(fields),
            "offset": offset,
            "limit": limit,
        }
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        retries = 3
        for attempt in range(retries):
            self._respect_rate_limit()
            try:
                resp = self._session.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=15,
                )
                self._last_request_time = time.time()
            except Exception as e:
                self.logger.error(f"Request error contacting Semantic Scholar: {e}")
                return None

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else max(1.0, 2 ** attempt)
                self.logger.warning(f"Rate limited by Semantic Scholar (429). Sleeping {sleep_seconds:.1f}s before retry...")
                time.sleep(sleep_seconds)
                continue

            if not resp.ok:
                self.logger.error(f"Semantic Scholar returned {resp.status_code}: {resp.text}")
                return None

            try:
                data = resp.json()
            except Exception as e:
                self.logger.error(f"Failed to parse Semantic Scholar response: {e}")
                return None

            return data.get("data", [])

        self.logger.error("Exceeded retry attempts for Semantic Scholar request")
        return None


def paper_search(query: str, max_results: int = 10, year_from: Optional[int] = None) -> str:
    """
    Synchronous wrapper for paper search.
    
    Args:
        query: Search query
        max_results: Maximum results to return
        year_from: Only return papers from this year onwards
        
    Returns:
        Formatted string with paper results
    """
    tool = PaperSearchTool(max_results=max_results)
    results = asyncio.run(tool.search(query, year_from=year_from))
    
    if not results:
        return "No academic papers found."
    
    # Format results as readable text
    output = f"Found {len(results)} academic papers for '{query}':\n\n"
    
    for i, paper in enumerate(results, 1):
        authors = ", ".join([a["name"] for a in paper["authors"][:3]])
        if len(paper["authors"]) > 3:
            authors += " et al."
            
        output += f"{i}. {paper['title']}\n"
        output += f"   Authors: {authors}\n"
        output += f"   Year: {paper['year']} | Citations: {paper['citation_count']}"
        if paper.get('venue'):
            output += f" | Venue: {paper['venue']}"
        output += "\n"
        
        if paper.get('abstract'):
            abstract = paper['abstract'][:200] + "..." if len(paper['abstract']) > 200 else paper['abstract']
            output += f"   Abstract: {abstract}\n"
            
        output += f"   URL: {paper['url']}\n"
        output += "\n"
    
    return output
