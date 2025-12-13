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
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation(config_path: str):
    """Run system evaluation."""
    import yaml
    from dotenv import load_dotenv
    from src.langgraph_orchestrator import LangGraphOrchestrator

    load_dotenv()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    orchestrator = LangGraphOrchestrator(config)
    
    test_query = "What are the key principles of accessible user interface design?"
    result = orchestrator.process_query(test_query)
    return result


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

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    orchestrator = LangGraphOrchestrator(config)

    while True:
        try:
            query = input("\nQuery: ").strip()

            if query.lower() in ["quit", "exit", "q"]:
                break

            if not query:
                continue

            result = orchestrator.process_query(query)

        except KeyboardInterrupt:
            break
        except Exception as e:
            continue

    return


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
