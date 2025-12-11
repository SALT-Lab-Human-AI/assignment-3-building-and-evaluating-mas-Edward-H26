"""
Safety Manager
Coordinates safety guardrails using NeMo Guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import os
import asyncio
from pathlib import Path

# Import custom actions for direct use when NeMo Rails is not available
from src.guardrails.nemo_config.actions import (
    detect_pii,
    redact_pii,
    check_harmful_content,
    check_factual_grounding,
    check_topic_relevance,
    detect_jailbreak,
    log_safety_event,
)


class SafetyManager:
    """
    Manages safety guardrails for the multi-agent system using NeMo Guardrails.

    Implements 5 safety categories:
    1. Jailbreak Prevention - Detect prompt injection and manipulation
    2. Content Moderation - Block harmful, violent, or offensive content
    3. Topic Relevance - Ensure queries are HCI-related
    4. PII Protection - Detect and redact personal information
    5. Factual Grounding - Flag potentially inaccurate claims
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize safety manager with NeMo Guardrails.

        Args:
            config: Safety configuration from config.yaml
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.log_events = config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log
        self.safety_events: List[Dict[str, Any]] = []

        # Prohibited categories
        self.prohibited_categories = config.get("prohibited_categories", [
            "harmful_content",
            "personal_attacks",
            "misinformation",
            "off_topic_queries"
        ])

        # Violation response strategy
        self.on_violation = config.get("on_violation", {})

        # Initialize NeMo Guardrails if available
        self.rails = None
        self.nemo_available = False

        if self.enabled:
            self._initialize_nemo_guardrails()

    def _initialize_nemo_guardrails(self):
        """Initialize NeMo Guardrails with config from nemo_config directory."""
        try:
            from nemoguardrails import LLMRails, RailsConfig

            # Path to NeMo config directory
            config_path = Path(__file__).parent / "nemo_config"

            if config_path.exists():
                self.logger.info(f"Loading NeMo Guardrails from {config_path}")
                rails_config = RailsConfig.from_path(str(config_path))
                self.rails = LLMRails(rails_config)
                self.nemo_available = True
                self.logger.info("NeMo Guardrails initialized successfully")
            else:
                self.logger.warning(
                    f"NeMo config path not found: {config_path}. "
                    "Using fallback safety checks."
                )

        except ImportError:
            self.logger.warning(
                "NeMo Guardrails not installed. Using fallback safety checks. "
                "Install with: pip install nemoguardrails"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize NeMo Guardrails: {e}")
            self.logger.info("Falling back to built-in safety checks")

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Check if input query is safe to process.

        Args:
            query: User query to check

        Returns:
            Dictionary with 'safe' boolean and optional 'violations' list
        """
        if not self.enabled:
            return {"safe": True}

        violations = []

        # Run safety checks (async wrapper for sync context)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    violations = pool.submit(
                        asyncio.run,
                        self._check_input_async(query)
                    ).result()
            else:
                violations = loop.run_until_complete(self._check_input_async(query))
        except RuntimeError:
            # No event loop, create new one
            violations = asyncio.run(self._check_input_async(query))

        is_safe = len(violations) == 0

        # Log safety event
        if self.log_events:
            self._log_safety_event("input", query, violations, is_safe)

        return {
            "safe": is_safe,
            "violations": violations
        }

    async def _check_input_async(self, query: str) -> List[Dict[str, Any]]:
        """
        Async input safety checks.

        Args:
            query: User query to check

        Returns:
            List of violations found
        """
        violations = []

        # 1. Jailbreak Detection
        if await detect_jailbreak(query):
            violations.append({
                "category": "jailbreak_attempt",
                "reason": "Query appears to be attempting to bypass safety guidelines",
                "severity": "high",
                "action": "blocked"
            })

        # 2. Topic Relevance Check
        if not await check_topic_relevance(query):
            # Only flag as violation if clearly off-topic
            off_topic_keywords = ["weather", "stock", "restaurant", "vacation", "recipe"]
            if any(kw in query.lower() for kw in off_topic_keywords):
                violations.append({
                    "category": "off_topic",
                    "reason": "Query does not appear to be related to HCI research",
                    "severity": "low",
                    "action": "redirect"
                })

        # 3. Harmful Content Check
        if await check_harmful_content(query):
            violations.append({
                "category": "harmful_content",
                "reason": "Query contains potentially harmful content",
                "severity": "high",
                "action": "blocked"
            })

        # 4. PII Detection in Input
        if await detect_pii(query):
            violations.append({
                "category": "pii_detected",
                "reason": "Query contains personal identifiable information",
                "severity": "medium",
                "action": "warning"
            })

        return violations

    def check_output_safety(self, response: str) -> Dict[str, Any]:
        """
        Check if output response is safe to return.

        Args:
            response: Generated response to check

        Returns:
            Dictionary with 'safe' boolean and modified 'response' if needed
        """
        if not self.enabled:
            return {"safe": True, "response": response}

        violations = []
        modified_response = response

        # Run safety checks
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run,
                        self._check_output_async(response)
                    ).result()
            else:
                result = loop.run_until_complete(self._check_output_async(response))

            violations = result["violations"]
            modified_response = result["response"]
        except RuntimeError:
            result = asyncio.run(self._check_output_async(response))
            violations = result["violations"]
            modified_response = result["response"]

        is_safe = len(violations) == 0

        # Log safety event
        if self.log_events and not is_safe:
            self._log_safety_event("output", response, violations, is_safe)

        # Apply violation handling
        final_response = modified_response
        if not is_safe:
            action = self.on_violation.get("action", "refuse")
            high_severity = any(v.get("severity") == "high" for v in violations)

            if high_severity and action == "refuse":
                final_response = self.on_violation.get(
                    "message",
                    "I cannot provide this response due to safety policies."
                )

        return {
            "safe": is_safe,
            "violations": violations,
            "response": final_response
        }

    async def _check_output_async(self, response: str) -> Dict[str, Any]:
        """
        Async output safety checks.

        Args:
            response: Generated response to check

        Returns:
            Dictionary with violations and potentially modified response
        """
        violations = []
        modified_response = response

        # 1. Harmful Content Check
        if await check_harmful_content(response):
            violations.append({
                "category": "harmful_content",
                "reason": "Response contains potentially harmful content",
                "severity": "high",
                "action": "blocked"
            })

        # 2. PII Detection and Redaction
        if await detect_pii(response):
            violations.append({
                "category": "pii_detected",
                "reason": "Response contains personal identifiable information",
                "severity": "medium",
                "action": "redacted"
            })
            # Redact PII from response
            modified_response = await redact_pii(modified_response)

        # 3. Factual Grounding Check
        if await check_factual_grounding(response):
            violations.append({
                "category": "factual_grounding",
                "reason": "Response contains claims that may need verification",
                "severity": "low",
                "action": "disclaimer_added"
            })
            # Add disclaimer for factual grounding
            modified_response += "\n\n[Note: Some claims in this response may benefit from verification with primary sources.]"

        return {
            "violations": violations,
            "response": modified_response
        }

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool
    ):
        """
        Log a safety event.

        Args:
            event_type: "input" or "output"
            content: The content that was checked
            violations: List of violations found
            is_safe: Whether content passed safety checks
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "categories": [v.get("category") for v in violations]
        }

        self.safety_events.append(event)

        if not is_safe:
            self.logger.warning(
                f"Safety event: {event_type} - safe={is_safe} - "
                f"categories={event['categories']}"
            )
        else:
            self.logger.debug(f"Safety check passed: {event_type}")

        # Write to safety log file if configured
        log_file = self.config.get("safety_log_file") or "logs/safety_events.log"
        if self.log_events:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                with open(log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Get all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """
        Get statistics about safety events.

        Returns:
            Dictionary with safety statistics
        """
        total = len(self.safety_events)
        input_events = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        violations = sum(1 for e in self.safety_events if not e["safe"])

        # Count by category
        category_counts = {}
        for event in self.safety_events:
            for category in event.get("categories", []):
                category_counts[category] = category_counts.get(category, 0) + 1

        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": violations / total if total > 0 else 0,
            "by_category": category_counts,
            "nemo_guardrails_active": self.nemo_available
        }

    def clear_events(self):
        """Clear safety event log."""
        self.safety_events = []

    def get_safety_categories(self) -> List[str]:
        """
        Get list of safety categories being checked.

        Returns:
            List of category names
        """
        return [
            "jailbreak_attempt",
            "off_topic",
            "harmful_content",
            "pii_detected",
            "factual_grounding"
        ]
