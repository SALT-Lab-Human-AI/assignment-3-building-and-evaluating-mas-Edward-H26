"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
  python main.py --mode langgraph     # Run LangGraph orchestrator (Novel Architecture)
"""

import argparse
import asyncio
import sys
from pathlib import Path


def run_cli(config_path: str):
    """Run CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main(argv=["--config", config_path])


def run_web():
    """Run web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation(config_path: str):
    """Run system evaluation."""
    import yaml
    from dotenv import load_dotenv
    from src.langgraph_orchestrator import LangGraphOrchestrator

    load_dotenv()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    print("Initializing LangGraph orchestrator...")
    orchestrator = LangGraphOrchestrator(config)
    
    # For now, run a simple test query
    # TODO: Integrate with SystemEvaluator for full evaluation
    print("\n" + "=" * 70)
    print("RUNNING TEST QUERY")
    print("=" * 70)
    
    test_query = "What are the key principles of accessible user interface design?"
    print(f"\nQuery: {test_query}\n")
    
    result = orchestrator.process_query(test_query)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nResponse:\n{result.get('response', 'No response generated')}")
    print(f"\nMetadata:")
    print(f"  - Messages: {result.get('metadata', {}).get('num_messages', 0)}")
    print(f"  - Sources: {result.get('metadata', {}).get('num_sources', 0)}")
    
    print("\n" + "=" * 70)
    print("Note: Full evaluation with SystemEvaluator can be implemented")
    print("=" * 70)


def run_langgraph(config_path: str):
    """
    Run LangGraph orchestrator with novel architecture.

    Features three innovations:
    1. Reflexion-Based Self-Correcting Agents
    2. Hierarchical Supervisor with Parallel Tool Execution
    3. Human-in-the-Loop Evaluation Triangulation
    """
    import yaml
    from dotenv import load_dotenv
    from src.langgraph_orchestrator import LangGraphOrchestrator

    # Load environment variables
    load_dotenv()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize LangGraph orchestrator
    print("\n" + "=" * 70)
    print("LANGGRAPH MULTI-AGENT HCI RESEARCH SYSTEM")
    print("=" * 70)
    print("\nNovel Architecture Innovations:")
    print("  1. Reflexion-Based Self-Correcting Agents")
    print("  2. Hierarchical Supervisor with Parallel Tool Execution")
    print("  3. Human-in-the-Loop Evaluation Triangulation")
    print("\nInitializing LangGraph orchestrator...")

    orchestrator = LangGraphOrchestrator(config)

    # Print workflow visualization
    print(orchestrator.visualize_workflow())

    # Interactive query loop
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter your HCI research queries (type 'quit' to exit):\n")

    while True:
        try:
            query = input("\nQuery: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting LangGraph orchestrator...")
                break

            if not query:
                continue

            print(f"\nProcessing: {query[:80]}...")
            print("-" * 70)

            result = orchestrator.process_query(query)

            # Display results
            print("\n" + "=" * 70)
            print("RESPONSE")
            print("=" * 70)
            response = result.get('response', 'No response generated')
            print(f"\n{response[:2000]}")
            if len(response) > 2000:
                print(f"\n... [truncated, {len(response)} total characters]")

            print("\n" + "-" * 70)
            print("WORKFLOW TRACE")
            print("-" * 70)
            for step in result.get("workflow_trace", []):
                step_name = step.get('step', 'Unknown')
                step_result = step.get('result', 'completed')
                print(f"  [{step_name}] {step_result}")

            print("\n" + "-" * 70)
            print("METADATA")
            print("-" * 70)
            meta = result.get("metadata", {})
            print(f"  Execution time: {meta.get('execution_time_seconds', 0):.2f}s")
            print(f"  Iterations: {meta.get('iterations', 0)}")
            print(f"  Paper sources: {meta.get('paper_sources', 0)}")
            print(f"  Web sources: {meta.get('web_sources', 0)}")
            print(f"  Reflexion score: {meta.get('reflexion_score', 0):.3f}")
            print(f"  LLM Judge score: {meta.get('llm_judge_score', 0):.3f}")
            print(f"  Decision: {meta.get('triangulated_decision', 'unknown')}")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Exiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

    # Print final statistics
    print("\n" + "=" * 70)
    print("SESSION STATISTICS")
    print("=" * 70)
    mem_stats = orchestrator.get_memory_statistics()
    safety_stats = orchestrator.get_safety_statistics()
    print(f"  Memory findings: {mem_stats.get('total_findings', 0)}")
    print(f"  Safety checks: {safety_stats.get('total_checks', 0)}")
    print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "langgraph"],
        default="langgraph",
        help="Mode to run: cli, web, evaluate, or langgraph (default)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(args.config)
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation(args.config))
    elif args.mode == "langgraph":
        run_langgraph(args.config)


if __name__ == "__main__":
    main()
