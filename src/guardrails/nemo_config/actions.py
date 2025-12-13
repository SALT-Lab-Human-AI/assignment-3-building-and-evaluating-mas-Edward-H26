"""
Custom Actions for NeMo Guardrails
These actions are called by the Colang rail definitions to perform
safety checks and content modifications.
"""

import re
import logging
from typing import Optional

# Try to import NeMo Guardrails action decorator
try:
    from nemoguardrails.actions import action
except ImportError:
    # Fallback decorator if nemoguardrails not installed
    def action(is_system_action: bool = False):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)


# PII Detection
# Common PII patterns
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone_us": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
    "ssn": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


@action(is_system_action=True)
async def detect_pii(response: str) -> bool:
    """
    Detect if response contains PII (Personally Identifiable Information).

    Args:
        response: The text to check for PII

    Returns:
        True if PII is detected, False otherwise
    """
    if not response:
        return False

    for pattern_name, pattern in PII_PATTERNS.items():
        if re.search(pattern, response, re.IGNORECASE):
            logger.warning(f"PII detected: {pattern_name}")
            return True

    return False


@action(is_system_action=True)
async def redact_pii(response: str) -> str:
    """
    Redact PII from response.

    Args:
        response: The text to redact PII from

    Returns:
        Text with PII redacted
    """
    if not response:
        return response

    redacted = response

    # Redact each type of PII
    redacted = re.sub(PII_PATTERNS["email"], "[EMAIL REDACTED]", redacted)
    redacted = re.sub(PII_PATTERNS["phone_us"], "[PHONE REDACTED]", redacted)
    redacted = re.sub(PII_PATTERNS["ssn"], "[SSN REDACTED]", redacted)
    redacted = re.sub(PII_PATTERNS["credit_card"], "[CARD REDACTED]", redacted)
    redacted = re.sub(PII_PATTERNS["ip_address"], "[IP REDACTED]", redacted)

    if redacted != response:
        logger.info("PII redacted from response")

    return redacted


# Harmful Content Detection
HARMFUL_KEYWORDS = [
    "hack", "exploit", "vulnerability", "attack", "malware",
    "phishing", "steal", "fraud", "illegal", "bypass security",
    "ddos", "injection", "xss", "sqli", "brute force",
]

OFFENSIVE_PATTERNS = [
    r"\b(hate|kill|destroy)\s+(all\s+)?\w+",
    r"\bdiscriminate\b",
    r"\bharass\b",
]


@action(is_system_action=True)
async def check_harmful_content(response: str) -> bool:
    """
    Check if response contains harmful content.

    Args:
        response: The text to check

    Returns:
        True if harmful content detected, False otherwise
    """
    if not response:
        return False

    response_lower = response.lower()

    # Check for harmful keywords in harmful context
    harmful_count = sum(1 for keyword in HARMFUL_KEYWORDS if keyword in response_lower)
    if harmful_count >= 3:  # Multiple harmful keywords suggest problematic content
        logger.warning(f"Potentially harmful content detected: {harmful_count} keywords")
        return True

    # Check for offensive patterns
    for pattern in OFFENSIVE_PATTERNS:
        if re.search(pattern, response_lower):
            logger.warning("Offensive pattern detected")
            return True

    return False


# Factual Grounding
SPECULATION_MARKERS = [
    "might be", "could be", "possibly", "perhaps",
    "I think", "I believe", "probably", "likely",
    "it seems", "appears to be", "may be", "uncertain",
]

STRONG_CLAIM_PATTERNS = [
    r"\b(always|never|all|none|every|best|worst)\b",
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bguarantee\b",
]


@action(is_system_action=True)
async def check_factual_grounding(response: str) -> bool:
    """
    Check if response contains claims that need factual grounding.

    Args:
        response: The text to check

    Returns:
        True if disclaimer should be added, False otherwise
    """
    if not response:
        return False

    response_lower = response.lower()

    # Count speculation markers
    speculation_count = sum(1 for marker in SPECULATION_MARKERS if marker in response_lower)

    # Check for strong claims without citations
    has_strong_claims = any(
        re.search(pattern, response_lower) for pattern in STRONG_CLAIM_PATTERNS
    )

    # Check if citations are present
    has_citations = bool(re.search(r"\[\w+,?\s*\d{4}\]|\(\w+,?\s*\d{4}\)", response))

    # If strong claims without citations, flag for disclaimer
    if has_strong_claims and not has_citations:
        logger.info("Strong claims without citations detected")
        return True

    # If high speculation without qualifications, flag
    if speculation_count >= 4:
        logger.info("High speculation level detected")
        return True

    return False


# Citation Check
@action(is_system_action=True)
async def check_citations(response: str) -> bool:
    """
    Check if response contains academic citations.

    Args:
        response: The text to check

    Returns:
        True if citations are present, False otherwise
    """
    if not response:
        return False

    # Common citation patterns
    citation_patterns = [
        r"\[\w+,?\s*\d{4}\]",  # [Author, 2023]
        r"\(\w+,?\s*\d{4}\)",  # (Author, 2023)
        r"\[\d+\]",            # [1]
        r"et al\.",            # et al.
        r"doi:\s*\S+",         # DOI
        r"https?://\S+",       # URLs as references
    ]

    for pattern in citation_patterns:
        if re.search(pattern, response, re.IGNORECASE):
            return True

    return False


# Response Quality
@action(is_system_action=True)
async def assess_response_quality(response: str) -> float:
    """
    Assess the quality of a response.

    Args:
        response: The text to assess

    Returns:
        Quality score between 0.0 and 1.0
    """
    if not response:
        return 0.0

    score = 0.5  # Base score

    # Length check (too short or too long is penalized)
    word_count = len(response.split())
    if word_count < 20:
        score -= 0.2
    elif word_count > 50:
        score += 0.1

    # Structure check (has paragraphs or lists)
    if "\n\n" in response or "\n-" in response or "\n*" in response:
        score += 0.1

    # Citation presence
    if await check_citations(response):
        score += 0.2

    # No harmful content
    if not await check_harmful_content(response):
        score += 0.1

    # Cap score at 1.0
    return min(1.0, max(0.0, score))


# Topic Relevance
HCI_KEYWORDS = [
    "hci", "human-computer", "user experience", "ux", "usability",
    "accessibility", "interface", "interaction design", "user research",
    "nielsen", "heuristics", "cognitive load", "fitts", "user centered",
    "design thinking", "prototyping", "wireframe", "user testing",
    "a/b testing", "eye tracking", "gesture", "voice ui", "vui",
    "mobile", "web design", "responsive", "wcag", "aria",
]


@action(is_system_action=True)
async def check_topic_relevance(query: str) -> bool:
    """
    Check if query is relevant to HCI topics.

    Args:
        query: The user query to check

    Returns:
        True if relevant to HCI, False otherwise
    """
    if not query:
        return False

    query_lower = query.lower()

    # Check for HCI-related keywords
    for keyword in HCI_KEYWORDS:
        if keyword in query_lower:
            return True

    # If no keywords found, might be off-topic
    return False


# Jailbreak Detection
JAILBREAK_PATTERNS = [
    r"ignore\s+(your\s+)?(previous\s+)?instructions",
    r"forget\s+(your\s+)?rules",
    r"pretend\s+(you\s+)?(are|have)",
    r"bypass\s+(your\s+)?safety",
    r"act\s+as\s+if",
    r"you\s+are\s+now\s+a",
    r"developer\s+mode",
    r"dan\s+mode",
    r"jailbreak",
    r"no\s+restrictions",
]


@action(is_system_action=True)
async def detect_jailbreak(query: str) -> bool:
    """
    Detect jailbreak attempts in user query.

    Args:
        query: The user query to check

    Returns:
        True if jailbreak attempt detected, False otherwise
    """
    if not query:
        return False

    query_lower = query.lower()

    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning(f"Jailbreak attempt detected: {pattern}")
            return True

    return False


# Safety Event Logging
def log_safety_event(
    event_type: str,
    content: str,
    is_blocked: bool,
    reason: Optional[str] = None
) -> dict:
    """
    Log a safety event for monitoring.

    Args:
        event_type: Type of event (input/output)
        content: The content that triggered the event
        is_blocked: Whether the content was blocked
        reason: Reason for blocking (if applicable)

    Returns:
        Dictionary with event details
    """
    import datetime

    event = {
        "timestamp": datetime.datetime.now().isoformat(),
        "type": event_type,
        "content_preview": content[:100] if content else "",
        "is_blocked": is_blocked,
        "reason": reason,
    }

    logger.info(f"Safety event: {event}")
    return event
