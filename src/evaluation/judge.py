"""
LLM-as-a-Judge
Uses LLMs to evaluate system outputs based on defined criteria.

Example usage:
    # Initialize judge with config
    judge = LLMJudge(config)
    
    # Evaluate a response
    result = await judge.evaluate(
        query="What is the capital of France?",
        response="Paris is the capital of France.",
        sources=[],
        ground_truth="Paris"
    )
    
    print(f"Overall Score: {result['overall_score']}")
    print(f"Criterion Scores: {result['criterion_scores']}")
"""

from typing import Dict, Any, List, Optional
import logging
import json
import os
from openai import OpenAI
from .rubrics import get_rubric_prompt, EVALUATION_RUBRICS


class LLMJudge:
    """
    LLM-based judge for evaluating system responses.

    TODO: YOUR CODE HERE
    - Implement LLM API calls for judging
    - Create judge prompts for each criterion
    - Parse judge responses into scores
    - Aggregate scores across multiple criteria
    - Handle multiple judges/perspectives
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM judge.

        Args:
            config: Configuration dictionary (from config.yaml)
        """
        self.config = config
        self.logger = logging.getLogger("evaluation.judge")

        # Load judge model configuration from config.yaml (models.judge)
        # This includes: provider, name, temperature, max_tokens
        self.model_config = config.get("models", {}).get("judge", {})

        # Load evaluation criteria from config.yaml (evaluation.criteria)
        # Each criterion has: name, weight, description
        self.criteria = config.get("evaluation", {}).get("criteria", [])
        
        # Initialize OpenAI client for LLM-as-a-Judge evaluation
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.logger.warning("OPENAI_API_KEY not found in environment")
        self.client = OpenAI(api_key=api_key) if api_key else None
        
        self.logger.info(f"LLMJudge initialized with {len(self.criteria)} criteria")
 
    async def evaluate(
        self,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a response using LLM-as-a-Judge.

        Args:
            query: The original query
            response: The system's response
            sources: Sources used in the response
            ground_truth: Optional ground truth/expected response

        Returns:
            Dictionary with scores for each criterion and overall score

        TODO: YOUR CODE HERE
        - Implement LLM API calls
        - Call judge for each criterion
        - Parse and aggregate scores
        - Provide detailed feedback
        """
        self.logger.info(f"Evaluating response for query: {query[:50]}...")

        results = {
            "query": query,
            "overall_score": 0.0,
            "criterion_scores": {},
            "feedback": [],
        }

        total_weight = sum(c.get("weight", 1.0) for c in self.criteria)
        weighted_score = 0.0

        # Evaluate each criterion
        for criterion in self.criteria:
            criterion_name = criterion.get("name", "unknown")
            weight = criterion.get("weight", 1.0)

            self.logger.info(f"Evaluating criterion: {criterion_name}")

            # TODO: Implement actual LLM judging
            score = await self._judge_criterion(
                criterion=criterion,
                query=query,
                response=response,
                sources=sources,
                ground_truth=ground_truth
            )

            results["criterion_scores"][criterion_name] = score
            weighted_score += score.get("score", 0.0) * weight

        # Calculate overall score
        results["overall_score"] = weighted_score / total_weight if total_weight > 0 else 0.0

        return results

    async def _judge_criterion(
        self,
        criterion: Dict[str, Any],
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> Dict[str, Any]:
        """
        Judge a single criterion.

        Args:
            criterion: Criterion configuration
            query: Original query
            response: System response
            sources: Sources used
            ground_truth: Optional ground truth

        Returns:
            Score and feedback for this criterion

        This is a basic implementation using Groq API.
        """
        criterion_name = criterion.get("name", "unknown")
        description = criterion.get("description", "")

        # Create judge prompt
        prompt = self._create_judge_prompt(
            criterion_name=criterion_name,
            description=description,
            query=query,
            response=response,
            sources=sources,
            ground_truth=ground_truth
        )

        # Call LLM API to get judgment
        try:
            judgment = await self._call_judge_llm(prompt)
            score_value, reasoning = self._parse_judgment(judgment)
            
            score = {
                "score": score_value,  # 0-1 scale
                "reasoning": reasoning,
                "criterion": criterion_name
            }
        except Exception as e:
            self.logger.error(f"Error judging criterion {criterion_name}: {e}")
            score = {
                "score": 0.0,
                "reasoning": f"Error during evaluation: {str(e)}",
                "criterion": criterion_name
            }

        return score

    def _create_judge_prompt(
        self,
        criterion_name: str,
        description: str,
        query: str,
        response: str,
        sources: Optional[List[Dict[str, Any]]],
        ground_truth: Optional[str]
    ) -> str:
        """
        Create a prompt for the judge LLM with detailed rubrics.

        Uses rubric-based scoring anchors from rubrics.py for consistent evaluation.
        """
        # Get detailed rubric for this criterion
        rubric_text = get_rubric_prompt(criterion_name)

        prompt = f"""You are an expert HCI research evaluator. Your task is to evaluate responses based on specific criteria with detailed rubrics.

# EVALUATION TASK

## Criterion: {criterion_name.replace('_', ' ').title()}
{description}

{rubric_text}

---

## Content to Evaluate

**Original Query:**
{query}

**Response to Evaluate:**
{response}
"""

        if sources:
            # Provide source details if available
            source_count = len(sources) if isinstance(sources, list) else 0
            prompt += f"\n**Sources Provided:** {source_count} sources"
            if source_count > 0 and isinstance(sources[0], dict):
                prompt += "\n"
                for i, s in enumerate(sources[:5], 1):
                    if isinstance(s, dict):
                        title = s.get('title', s.get('name', 'Unknown'))
                        prompt += f"  {i}. {title}\n"

        # For evidence_quality, analyze inline citations in the response
        if criterion_name == "evidence_quality":
            import re
            # Count inline citations [Author, Year] format
            citation_pattern = r'\[([A-Z][a-z]+(?:\s+(?:et al\.|&\s+[A-Z][a-z]+))?),?\s*\d{4}\]'
            citations = re.findall(citation_pattern, response)
            citation_count = len(citations)
            unique_citations = list(set(citations))

            prompt += f"\n\n**Citation Analysis (Auto-detected):**"
            prompt += f"\n- Total inline citations found: {citation_count}"
            prompt += f"\n- Unique sources cited: {len(unique_citations)}"
            if unique_citations:
                prompt += f"\n- Citation examples: {unique_citations[:5]}"

            # Check for References section
            has_references = "## References" in response or "References" in response
            prompt += f"\n- Has References section: {'Yes' if has_references else 'No'}"

            # Provide explicit scoring guidance based on citation count
            prompt += "\n\n**MANDATORY SCORING GUIDE (follow exactly):**"
            if citation_count >= 10 and len(unique_citations) >= 6:
                prompt += f"\n- Found {citation_count} citations with {len(unique_citations)} unique sources → SCORE = 1.0"
            elif citation_count >= 8 and len(unique_citations) >= 5:
                prompt += f"\n- Found {citation_count} citations with {len(unique_citations)} unique sources → SCORE = 0.9"
            elif citation_count >= 6 and len(unique_citations) >= 4:
                prompt += f"\n- Found {citation_count} citations with {len(unique_citations)} unique sources → SCORE = 0.85"
            elif citation_count >= 5 and len(unique_citations) >= 3:
                prompt += f"\n- Found {citation_count} citations with {len(unique_citations)} unique sources → SCORE = 0.8"
            elif citation_count >= 3:
                prompt += f"\n- Found {citation_count} citations with {len(unique_citations)} unique sources → SCORE = 0.7"
            else:
                prompt += f"\n- Found {citation_count} citations with {len(unique_citations)} unique sources → SCORE = 0.6 or lower"

            prompt += "\n\nYOU MUST USE THE SCORE FROM THE MANDATORY SCORING GUIDE ABOVE. Do not deviate."

        # For factual_accuracy, add guidance about cited claims
        if criterion_name == "factual_accuracy":
            import re
            # Count inline citations
            citation_pattern = r'\[([A-Z][a-z]+(?:\s+(?:et al\.|&\s+[A-Z][a-z]+))?),?\s*\d{4}\]'
            citations = re.findall(citation_pattern, response)
            citation_count = len(citations)

            prompt += f"\n\n**IMPORTANT FACTUAL ACCURACY SCORING GUIDANCE:**"
            prompt += f"\n- Claims supported by inline citations [Author, Year] should be treated as VERIFIED"
            prompt += f"\n- This response has {citation_count} inline citations"
            prompt += f"\n- Only flag claims as 'unverifiable' if they have NO citation support"
            prompt += f"\n- If most claims have citation support, score should be >= 0.85"
            prompt += f"\n- DO NOT penalize for 'unverifiable claims' if those claims have proper citations"

        if ground_truth:
            prompt += f"\n**Ground Truth / Expected Answer:**\n{ground_truth}"

        prompt += """

---

## EVALUATION INSTRUCTIONS

1. Carefully read the response and compare against the rubric anchors
2. Identify which score anchor (1.0, 0.9, 0.8, etc.) best matches the response quality
3. Consider the evaluation questions when making your assessment
4. Be objective and consistent - use the rubric anchors as your primary guide

## OUTPUT FORMAT (JSON only)

Respond with ONLY valid JSON in this exact format:
```json
{
    "score": <float between 0.0 and 1.0 matching rubric anchors>,
    "reasoning": "<2-3 sentences explaining your score based on the rubric>",
    "rubric_match": "<which rubric anchor score (e.g., '0.8') best matches>"
}
```
"""

        return prompt

    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call LLM API to get judgment.
        Uses model configuration from config.yaml (models.judge section).
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check OPENAI_API_KEY environment variable.")

        try:
            # Load model settings from config.yaml (models.judge)
            model_name = self.model_config.get("name", "gpt-4o-mini")
            temperature = self.model_config.get("temperature", 0.3)
            max_tokens = self.model_config.get("max_tokens", 1024)

            self.logger.debug(f"Calling OpenAI API with model: {model_name}")

            # Build API parameters - handle different model requirements
            # o1 and o3 models require max_completion_tokens instead of max_tokens
            api_params = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert evaluator. Provide your evaluations in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": model_name,
            }

            # Check if model requires max_completion_tokens (o1, o3 models)
            if model_name.startswith("o1") or model_name.startswith("o3"):
                api_params["max_completion_tokens"] = max_tokens
                # o1/o3 models don't support temperature parameter
            else:
                api_params["max_tokens"] = max_tokens
                api_params["temperature"] = temperature

            # Call OpenAI API for evaluation
            chat_completion = self.client.chat.completions.create(**api_params)
            
            response = chat_completion.choices[0].message.content
            self.logger.debug(f"Received response: {response[:100]}...")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API: {e}")
            raise

    def _parse_judgment(self, judgment: str) -> tuple:
        """
        Parse LLM judgment response.
        
        """
        try:
            # Clean up the response - remove markdown code blocks if present
            judgment_clean = judgment.strip()
            if judgment_clean.startswith("```json"):
                judgment_clean = judgment_clean[7:]
            elif judgment_clean.startswith("```"):
                judgment_clean = judgment_clean[3:]
            if judgment_clean.endswith("```"):
                judgment_clean = judgment_clean[:-3]
            judgment_clean = judgment_clean.strip()
            
            # Parse JSON
            result = json.loads(judgment_clean)
            score = float(result.get("score", 0.0))
            reasoning = result.get("reasoning", "")
            
            # Validate score is in range [0, 1]
            score = max(0.0, min(1.0, score))
            
            return score, reasoning
            
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Raw judgment: {judgment[:200]}")
            return 0.0, f"Error parsing judgment: Invalid JSON"
        except Exception as e:
            self.logger.error(f"Error parsing judgment: {e}")
            return 0.0, f"Error parsing judgment: {str(e)}"



async def example_basic_evaluation():
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    judge = LLMJudge(config)

    query = "What is the capital of France?"
    response = "Paris is the capital of France. It is known for the Eiffel Tower."
    ground_truth = "Paris"

    result = await judge.evaluate(
        query=query,
        response=response,
        sources=[],
        ground_truth=ground_truth
    )


async def example_compare_responses():
    import yaml
    from dotenv import load_dotenv

    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    judge = LLMJudge(config)

    query = "What causes climate change?"
    ground_truth = "Climate change is primarily caused by increased greenhouse gas emissions from human activities, including burning fossil fuels, deforestation, and industrial processes."

    responses = [
        "Climate change is primarily caused by greenhouse gas emissions from human activities.",
        "The weather changes because of natural cycles and the sun's activity.",
        "Climate change is a complex phenomenon involving multiple factors including CO2 emissions, deforestation, and industrial processes."
    ]

    results = []
    for i, response in enumerate(responses, 1):
        result = await judge.evaluate(
            query=query,
            response=response,
            sources=[],
            ground_truth=ground_truth
        )
        results.append(result)

    best_idx = max(range(len(results)), key=lambda i: results[i]["overall_score"])


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_basic_evaluation())
    asyncio.run(example_compare_responses())
