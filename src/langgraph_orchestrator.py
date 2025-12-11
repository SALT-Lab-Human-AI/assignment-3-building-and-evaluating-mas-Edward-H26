"""
LangGraph Orchestrator - Main Entry Point

Orchestrates the Multi-Agent HCI Research System using LangGraph.

Three Novel Innovations:
1. Reflexion-Based Self-Correcting Agents - Agents learn from past failures
2. Hierarchical Supervisor with Parallel Tool Execution - Efficient tool calling
3. Human-in-the-Loop Evaluation Triangulation - Three-source decision making
"""

import logging
from typing import Dict, Any, List, Optional
import yaml
from datetime import datetime

from src.langgraph import (
    create_supervisor_graph,
    compile_graph,
    create_initial_state,
    SafetyGuardian,
    ReflexionEngine,
)
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
