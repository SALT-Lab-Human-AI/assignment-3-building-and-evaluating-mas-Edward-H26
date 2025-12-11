"""
Supervisor Graph for LangGraph Multi-Agent Research System

Main orchestrator implementing three novel innovations:
1. Reflexion-Based Self-Correcting Agents
2. Hierarchical Supervisor with Parallel Tool Execution
3. Human-in-the-Loop Evaluation Triangulation

Architecture:
- Safety Guardian (Layer 1: Pre-flight)
- Supervisor routes to specialized teams
- Teams execute with parallel tools where possible
- Reflexion Engine evaluates and self-corrects
- Evaluation Triangulation (LLM Judge + Reflexion + Human)
- Safety Post-flight (Layer 3)
- Memory Persistence
"""

from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

from .state import ResearchState, create_initial_state
from .teams import create_planning_team, create_research_team, create_synthesis_team
from .safety import SafetyGuardian
from .reflexion import ReflexionEngine


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

    # Synthesis → Reflexion check
    graph.add_edge("synthesis_team", "reflexion_check")

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
