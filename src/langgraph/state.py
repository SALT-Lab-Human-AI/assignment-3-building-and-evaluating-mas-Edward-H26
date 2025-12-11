"""
LangGraph State Schema for Multi-Agent Research System

Defines comprehensive typed state for the research workflow including:
- Reflexion memory for self-correction
- Team outputs for hierarchical coordination
- Human feedback for evaluation triangulation
"""

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
