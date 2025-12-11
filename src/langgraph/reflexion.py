"""
Reflexion Engine for Self-Correcting Agents

Implements the Reflexion pattern where agents:
1. Generate output
2. Self-evaluate quality
3. Learn from mistakes via persistent memory
4. Apply lessons to future queries
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os


class ReflexionEngine:
    """
    Reflexion-based self-correction engine.

    Maintains memory of past successes/failures and applies learned
    lessons to improve future performance.
    """

    def __init__(self, memory: Optional[Dict] = None):
        self.memory = memory or {
            "past_failures": [],
            "successful_patterns": [],
            "tool_effectiveness": {},
        }

    def evaluate(
        self,
        query: str,
        draft: str,
        sources: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate output quality and generate improvement feedback.

        Returns:
            {
                "score": float (0-1),
                "feedback": str,
                "lesson": str (for memory),
                "issues": List[str]
            }
        """
        issues = []
        score = 0.0

        if not draft or len(draft) < 100:
            issues.append("Response too brief")
            score -= 0.3
        else:
            score += 0.3

        if not sources:
            issues.append("No sources provided")
            score -= 0.2
        elif len(sources) < 3:
            issues.append("Insufficient sources (< 3)")
            score -= 0.1
        else:
            score += 0.3

        citation_markers = ["[", "(", "et al", "202"]
        has_citations = any(m in draft for m in citation_markers)
        if not has_citations:
            issues.append("Missing inline citations")
            score -= 0.2
        else:
            score += 0.2

        query_words = set(query.lower().split())
        draft_words = set(draft.lower().split())
        overlap = len(query_words & draft_words)
        if overlap < len(query_words) * 0.3:
            issues.append("Response may not address query directly")
            score -= 0.1
        else:
            score += 0.2

        past_lessons = self._get_relevant_lessons(query)
        if past_lessons:
            for lesson in past_lessons:
                if lesson.get("error_type", "") in str(issues):
                    issues.append(f"Repeated past error: {lesson.get('lesson', '')}")
                    score -= 0.1

        score = max(0.0, min(1.0, score + 0.5))

        feedback = self._generate_feedback(issues, score)
        lesson = self._extract_lesson(query, issues, score)

        return {
            "score": score,
            "feedback": feedback,
            "lesson": lesson,
            "issues": issues,
        }

    def update_memory(
        self,
        query: str,
        success: bool,
        lesson: str
    ) -> Dict[str, Any]:
        """Update reflexion memory with new insights."""
        entry = {
            "query": query[:100],
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "lesson": lesson,
            "error_type": "quality" if not success else "none",
        }

        if success:
            self.memory["successful_patterns"].append(entry)
            if len(self.memory["successful_patterns"]) > 20:
                self.memory["successful_patterns"] = self.memory["successful_patterns"][-20:]
        else:
            self.memory["past_failures"].append(entry)
            if len(self.memory["past_failures"]) > 20:
                self.memory["past_failures"] = self.memory["past_failures"][-20:]

        return self.memory

    def update_tool_effectiveness(self, tool: str, success: bool) -> None:
        """Track tool effectiveness for adaptive selection."""
        if tool not in self.memory["tool_effectiveness"]:
            self.memory["tool_effectiveness"][tool] = {"total": 0, "success": 0}

        self.memory["tool_effectiveness"][tool]["total"] += 1
        if success:
            self.memory["tool_effectiveness"][tool]["success"] += 1

    def get_tool_score(self, tool: str) -> float:
        """Get effectiveness score for a tool."""
        if tool not in self.memory["tool_effectiveness"]:
            return 0.5
        data = self.memory["tool_effectiveness"][tool]
        if data["total"] == 0:
            return 0.5
        return data["success"] / data["total"]

    def _get_relevant_lessons(self, query: str) -> List[Dict]:
        """Find past lessons relevant to current query."""
        query_words = set(query.lower().split())
        relevant = []

        for failure in self.memory.get("past_failures", []):
            past_words = set(failure.get("query", "").lower().split())
            overlap = len(query_words & past_words)
            if overlap >= 2:
                relevant.append(failure)

        return relevant[:3]

    def _generate_feedback(self, issues: List[str], score: float) -> str:
        """Generate actionable feedback."""
        if score >= 0.7:
            return "Output quality is acceptable. Minor improvements possible."
        elif score >= 0.5:
            feedback = "Output needs improvement:\n"
            for i, issue in enumerate(issues, 1):
                feedback += f"{i}. {issue}\n"
            return feedback
        else:
            return f"Output quality is poor. Critical issues: {', '.join(issues)}"

    def _extract_lesson(self, query: str, issues: List[str], score: float) -> str:
        """Extract a lesson for future reference."""
        if score >= 0.7:
            return f"Successful approach for query type: {query[:30]}..."
        elif issues:
            return f"Avoid: {issues[0]} for queries like: {query[:30]}..."
        return ""

    def save_to_file(self, filepath: str) -> None:
        """Persist reflexion memory."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
        with open(filepath, "w") as f:
            json.dump(self.memory, f, indent=2)

    def load_from_file(self, filepath: str) -> bool:
        """Load reflexion memory."""
        try:
            with open(filepath, "r") as f:
                self.memory = json.load(f)
            return True
        except Exception:
            return False
