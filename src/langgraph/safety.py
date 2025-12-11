"""
SafetyGuardian: 3-Layer Safety System for LangGraph

Implements a comprehensive safety architecture:
- Layer 1: Pre-flight (input validation, jailbreak detection)
- Layer 2: In-flight (tool output validation, PII detection)
- Layer 3: Post-flight (output sanitization, harmful content check)
"""

from typing import Dict, Any, List
import re
from datetime import datetime


class SafetyGuardian:
    """
    3-Layer Safety System for LangGraph.

    Provides comprehensive safety checks at every stage of the research pipeline.
    """

    def __init__(self):
        self.safety_log: List[Dict] = []

        self.hci_keywords = [
            "hci", "user interface", "usability", "accessibility",
            "user experience", "ux", "interaction design", "human factors",
            "interface", "ergonomics", "user study", "chi", "uist",
            "human-computer", "design", "prototype", "evaluation",
            "nielsen", "heuristics", "cognitive", "mental model"
        ]

        self.jailbreak_patterns = [
            r"ignore.*instructions",
            r"pretend.*you.*are",
            r"act.*as.*if",
            r"bypass.*safety",
            r"override.*rules",
            r"disregard.*previous",
            r"forget.*everything",
        ]

        self.harmful_patterns = [
            r"how.*to.*hack",
            r"how.*to.*harm",
            r"illegal.*activity",
            r"violence",
            r"exploit.*vulnerability",
            r"steal.*data",
            r"malware",
        ]

    def preflight_check(self, query: str, context: str = "") -> Dict[str, Any]:
        """Layer 1: Validate input before processing."""
        violations = []

        query_lower = query.lower()

        for pattern in self.jailbreak_patterns:
            if re.search(pattern, query_lower):
                violations.append({
                    "layer": 1,
                    "type": "jailbreak",
                    "severity": "critical",
                    "message": "Potential jailbreak attempt detected"
                })
                break

        for pattern in self.harmful_patterns:
            if re.search(pattern, query_lower):
                violations.append({
                    "layer": 1,
                    "type": "harmful_input",
                    "severity": "critical",
                    "message": "Harmful content detected"
                })
                break

        has_hci = any(kw in query_lower for kw in self.hci_keywords)
        if not has_hci and len(query.split()) > 5:
            violations.append({
                "layer": 1,
                "type": "off_topic",
                "severity": "warning",
                "message": "Query may not be HCI-related"
            })

        can_proceed = not any(v["severity"] == "critical" for v in violations)

        self._log("preflight", query[:100], can_proceed, violations)

        return {
            "can_proceed": can_proceed,
            "violations": violations,
            "safe": can_proceed,
        }

    def inflight_check(self, tool_output: str, tool_name: str) -> Dict[str, Any]:
        """Layer 2: Validate tool outputs during execution."""
        violations = []
        sanitized = tool_output

        if self._detect_pii(tool_output):
            sanitized = self._redact_pii(tool_output)
            violations.append({
                "layer": 2,
                "type": "pii_detected",
                "severity": "warning",
                "message": f"PII detected in {tool_name} results"
            })

        if len(tool_output.strip()) < 50:
            violations.append({
                "layer": 2,
                "type": "insufficient_results",
                "severity": "info",
                "message": f"{tool_name} returned minimal results"
            })

        self._log("inflight", tool_output[:100], True, violations)

        return {
            "safe": True,
            "violations": violations,
            "sanitized_output": sanitized,
        }

    def postflight_check(self, response: str) -> Dict[str, Any]:
        """Layer 3: Validate and sanitize final output."""
        violations = []
        sanitized = response

        if self._detect_pii(response):
            sanitized = self._redact_pii(response)
            violations.append({
                "layer": 3,
                "type": "pii_redacted",
                "severity": "info",
                "message": "PII redacted from response"
            })

        for pattern in self.harmful_patterns:
            if re.search(pattern, response.lower()):
                violations.append({
                    "layer": 3,
                    "type": "harmful_output",
                    "severity": "critical",
                    "message": "Harmful content in response"
                })
                break

        if len(response) < 200:
            violations.append({
                "layer": 3,
                "type": "insufficient_response",
                "severity": "warning",
                "message": "Response may be too brief"
            })

        safe = not any(v["severity"] == "critical" for v in violations)

        self._log("postflight", response[:100], safe, violations)

        return {
            "safe": safe,
            "violations": violations,
            "sanitized_response": sanitized if safe else self._get_blocked_message(),
        }

    def _detect_pii(self, text: str) -> bool:
        """Detect personally identifiable information."""
        patterns = [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            r"\b\d{3}-\d{2}-\d{4}\b",
        ]
        return any(re.search(p, text) for p in patterns)

    def _redact_pii(self, text: str) -> str:
        """Redact PII from text."""
        text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL]",
            text
        )
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
        return text

    def _get_blocked_message(self) -> str:
        """Get message for blocked responses."""
        return "I cannot process this request due to safety guidelines. Please ask a question related to Human-Computer Interaction research."

    def _log(self, layer: str, content: str, safe: bool, violations: List) -> None:
        """Log safety event."""
        self.safety_log.append({
            "timestamp": datetime.now().isoformat(),
            "layer": layer,
            "content_preview": content,
            "safe": safe,
            "violations": violations,
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics."""
        total = len(self.safety_log)
        by_layer = {"preflight": 0, "inflight": 0, "postflight": 0}
        violations_by_type: Dict[str, int] = {}

        for event in self.safety_log:
            layer = event.get("layer", "unknown")
            by_layer[layer] = by_layer.get(layer, 0) + 1
            for v in event.get("violations", []):
                vtype = v.get("type", "unknown")
                violations_by_type[vtype] = violations_by_type.get(vtype, 0) + 1

        return {
            "total_checks": total,
            "by_layer": by_layer,
            "violations_by_type": violations_by_type,
        }

    def clear_log(self) -> None:
        """Clear safety log."""
        self.safety_log = []
