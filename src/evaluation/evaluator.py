"""
System Evaluator
Runs batch evaluations and generates reports.

Example usage:
    # Load config
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator with orchestrator
    evaluator = SystemEvaluator(config, orchestrator=my_orchestrator)
    
    # Run evaluation
    report = await evaluator.evaluate_system("data/test_queries.json")
    
    # Results are automatically saved to outputs/
"""

from typing import Dict, Any, List, Optional
import json
import logging
from pathlib import Path
from datetime import datetime
import asyncio

from .judge import LLMJudge


class SystemEvaluator:
    """
    Evaluates the multi-agent system using test queries and LLM-as-a-Judge.

    TODO: YOUR CODE HERE
    - Load test queries from file
    - Run system on all test queries
    - Collect and aggregate results
    - Generate evaluation report
    - Perform error analysis
    """

    def __init__(self, config: Dict[str, Any], orchestrator=None):
        """
        Initialize evaluator.

        Args:
            config: Configuration dictionary (from config.yaml)
            orchestrator: The orchestrator to evaluate
        """
        self.config = config
        self.orchestrator = orchestrator
        self.logger = logging.getLogger("evaluation.evaluator")

        # Load evaluation configuration from config.yaml
        eval_config = config.get("evaluation", {})
        self.enabled = eval_config.get("enabled", True)
        self.max_test_queries = eval_config.get("num_test_queries", None)
        
        # Initialize judge (passes config to load judge model settings and criteria)
        self.judge = LLMJudge(config)

        # Evaluation results
        self.results: List[Dict[str, Any]] = []
        
        self.logger.info(f"SystemEvaluator initialized (enabled={self.enabled})")

    async def evaluate_system(
        self,
        test_queries_path: str = "data/test_queries.json"
    ) -> Dict[str, Any]:
        """
        Run full system evaluation.

        Args:
            test_queries_path: Path to test queries JSON file

        Returns:
            Evaluation results and statistics

        TODO: YOUR CODE HERE
        - Load test queries
        - Run system on each query
        - Evaluate each response
        - Aggregate results
        - Generate report
        """
        # Check if evaluation is enabled in config.yaml
        if not self.enabled:
            self.logger.warning("Evaluation is disabled in config.yaml")
            return {"error": "Evaluation is disabled in configuration"}
        
        self.logger.info("Starting system evaluation")

        # Load test queries
        test_queries = self._load_test_queries(test_queries_path)
        self.logger.info(f"Loaded {len(test_queries)} test queries")

        # Evaluate each query
        for i, test_case in enumerate(test_queries, 1):
            self.logger.info(f"Evaluating query {i}/{len(test_queries)}")

            try:
                result = await self._evaluate_query(test_case)
                self.results.append(result)
            except Exception as e:
                self.logger.error(f"Error evaluating query {i}: {e}")
                self.results.append({
                    "query": test_case.get("query", ""),
                    "error": str(e)
                })

        # Aggregate results
        report = self._generate_report()

        # Save results
        self._save_results(report)

        return report

    async def _evaluate_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single test query.

        Args:
            test_case: Test case with query and optional ground truth

        Returns:
            Evaluation result for this query

        This shows how to integrate with the orchestrator.
        """
        query = test_case.get("query", "")
        ground_truth = test_case.get("ground_truth")
        expected_sources = test_case.get("expected_sources", [])

        # Run through orchestrator if available
        if self.orchestrator:
            try:
                # Call orchestrator's process_query method
                # TODO: YOUR CODE HERE
                # Need to implement this in their orchestrator
                response_data = self.orchestrator.process_query(query)
                
                # If process_query is async, use:
                # response_data = await self.orchestrator.process_query(query)
                
            except Exception as e:
                self.logger.error(f"Error processing query through orchestrator: {e}")
                response_data = {
                    "query": query,
                    "response": f"Error: {str(e)}",
                    "citations": [],
                    "metadata": {"error": str(e)}
                }
        else:
            # Placeholder for testing without orchestrator
            self.logger.warning("No orchestrator provided, using placeholder response")
            response_data = {
                "query": query,
                "response": "Placeholder response - orchestrator not connected",
                "citations": [],
                "metadata": {"num_sources": 0}
            }

        # Evaluate response using LLM-as-a-Judge
        evaluation = await self.judge.evaluate(
            query=query,
            response=response_data.get("response", ""),
            sources=response_data.get("metadata", {}).get("sources", []),
            ground_truth=ground_truth
        )

        return {
            "query": query,
            "response": response_data.get("response", ""),
            "evaluation": evaluation,
            "metadata": response_data.get("metadata", {}),
            "ground_truth": ground_truth
        }

    def _load_test_queries(self, path: str) -> List[Dict[str, Any]]:
        """
        Load test queries from JSON file.

        TODO: YOUR CODE HERE
        - Create test query dataset
        - Load and validate queries
        """
        path_obj = Path(path)
        if not path_obj.exists():
            self.logger.warning(f"Test queries file not found: {path}")
            return []

        with open(path_obj, 'r') as f:
            queries = json.load(f)

        # Limit number of queries if configured in config.yaml
        if self.max_test_queries and len(queries) > self.max_test_queries:
            self.logger.info(f"Limiting to {self.max_test_queries} queries (from config.yaml)")
            queries = queries[:self.max_test_queries]

        return queries

    def _generate_report(self) -> Dict[str, Any]:
        """
        Generate evaluation report with statistics and analysis.

        TODO: YOUR CODE HERE
        - Calculate aggregate statistics
        - Identify best/worst performing queries
        - Analyze errors
        - Generate visualizations (optional)
        """
        if not self.results:
            return {"error": "No results to report"}

        # Calculate statistics
        total_queries = len(self.results)
        successful = [r for r in self.results if "error" not in r]
        failed = [r for r in self.results if "error" in r]

        # Aggregate scores
        criterion_scores = {}
        overall_scores = []

        for result in successful:
            evaluation = result.get("evaluation", {})
            overall_scores.append(evaluation.get("overall_score", 0.0))

            # Collect scores by criterion
            for criterion, score_data in evaluation.get("criterion_scores", {}).items():
                if criterion not in criterion_scores:
                    criterion_scores[criterion] = []
                criterion_scores[criterion].append(score_data.get("score", 0.0))

        # Calculate averages
        avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

        avg_criterion_scores = {}
        for criterion, scores in criterion_scores.items():
            avg_criterion_scores[criterion] = sum(scores) / len(scores) if scores else 0.0

        # Find best and worst
        best_result = max(successful, key=lambda r: r.get("evaluation", {}).get("overall_score", 0.0)) if successful else None
        worst_result = min(successful, key=lambda r: r.get("evaluation", {}).get("overall_score", 0.0)) if successful else None

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_queries": total_queries,
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / total_queries if total_queries > 0 else 0.0
            },
            "scores": {
                "overall_average": avg_overall,
                "by_criterion": avg_criterion_scores
            },
            "best_result": {
                "query": best_result.get("query", "") if best_result else "",
                "score": best_result.get("evaluation", {}).get("overall_score", 0.0) if best_result else 0.0
            } if best_result else None,
            "worst_result": {
                "query": worst_result.get("query", "") if worst_result else "",
                "score": worst_result.get("evaluation", {}).get("overall_score", 0.0) if worst_result else 0.0
            } if worst_result else None,
            "detailed_results": self.results
        }

        return report

    def _save_results(self, report: Dict[str, Any]):
        """
        Save evaluation results to file.

        TODO: YOUR CODE HERE
        - Save detailed results
        - Generate visualizations
        - Create summary report
        """
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"evaluation_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Evaluation results saved to {results_file}")

        # Save summary
        summary_file = output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            summary = report.get("summary", {})
            f.write(f"Total Queries: {summary.get('total_queries', 0)}\n")
            f.write(f"Successful: {summary.get('successful', 0)}\n")
            f.write(f"Failed: {summary.get('failed', 0)}\n")
            f.write(f"Success Rate: {summary.get('success_rate', 0.0):.2%}\n\n")

            scores = report.get("scores", {})
            f.write(f"Overall Average Score: {scores.get('overall_average', 0.0):.3f}\n\n")

            f.write("Scores by Criterion:\n")
            for criterion, score in scores.get("by_criterion", {}).items():
                f.write(f"  {criterion}: {score:.3f}\n")

        self.logger.info(f"Summary saved to {summary_file}")

    def export_for_report(self, output_path: str = "outputs/report_data.json"):
        """
        Export data formatted for inclusion in technical report.

        """
        if not self.results:
            self.logger.warning("No results to export")
            return
        
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)
        
        # Format data for report
        report_data = {
            "evaluation_date": datetime.now().isoformat(),
            "total_queries": len(self.results),
            "results": self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Report data exported to {output_path}")


async def example_simple_evaluation():
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    test_queries = [
        {
            "query": "What is the capital of France?",
            "ground_truth": "Paris is the capital of France."
        },
        {
            "query": "What are the benefits of exercise?",
            "ground_truth": "Exercise improves physical health, mental wellbeing, and reduces disease risk."
        }
    ]

    test_file = Path("data/test_queries_example.json")
    test_file.parent.mkdir(exist_ok=True)
    with open(test_file, "w") as f:
        json.dump(test_queries, f, indent=2)

    evaluator = SystemEvaluator(config, orchestrator=None)

    report = await evaluator.evaluate_system(str(test_file))

    test_file.unlink()


async def example_with_orchestrator():
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # TODO: YOUR CODE HERE
    from src.autogen_orchestrator import LangGraphOrchestrator
    orchestrator = LangGraphOrchestrator(config)

    test_queries = [
        {
            "query": "What are the key principles of accessible user interface design?",
            "ground_truth": "Key principles include perceivability, operability, understandability, and robustness."
        }
    ]

    test_file = Path("data/test_queries_orchestrator.json")
    test_file.parent.mkdir(exist_ok=True)
    with open(test_file, "w") as f:
        json.dump(test_queries, f, indent=2)

    evaluator = SystemEvaluator(config, orchestrator=orchestrator)

    report = await evaluator.evaluate_system(str(test_file))

    test_file.unlink()


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_simple_evaluation())
    asyncio.run(example_with_orchestrator())
