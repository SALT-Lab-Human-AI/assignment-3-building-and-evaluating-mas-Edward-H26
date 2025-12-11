"""
LangGraph Multi-Agent Research System

Novel architecture with three innovations:
1. Reflexion-Based Self-Correcting Agents
2. Hierarchical Supervisor with Parallel Tool Execution
3. Human-in-the-Loop Evaluation Triangulation
"""

from .state import ResearchState, ReflexionMemory, TeamOutput, create_initial_state
from .reflexion import ReflexionEngine
from .safety import SafetyGuardian
from .teams import create_planning_team, create_research_team, create_synthesis_team
from .supervisor import create_supervisor_graph, compile_graph

__all__ = [
    # State
    "ResearchState",
    "ReflexionMemory",
    "TeamOutput",
    "create_initial_state",
    # Core Components
    "ReflexionEngine",
    "SafetyGuardian",
    # Teams
    "create_planning_team",
    "create_research_team",
    "create_synthesis_team",
    # Supervisor
    "create_supervisor_graph",
    "compile_graph",
]
