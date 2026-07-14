from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache


SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
        re.I,
    ),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,255}|"
        r"github_pat_[A-Za-z0-9_]{20,255})\b"
    ),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "jwt": re.compile(
        r"\beyJ[A-Za-z0-9_-]{10,}\."
        r"[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
    ),
    "password_assignment": re.compile(
        r"(?i)\b(?:password|passwd|pwd|api[_ -]?key|secret|token)\b"
        r"\s*[:=]\s*['\"]?[^\s,'\"]{8,}"
    ),
}

IDENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "passport": re.compile(
        r"(?i)\bpassport\b|machine readable zone|"
        r"nationality.*date of birth"
    ),
    "driver_license": re.compile(
        r"(?i)driver'?s? licen[cs]e|learner'?s? permit"
    ),
    "national_id": re.compile(
        r"(?i)national identification|national id card|identity card"
    ),
    "social_security_card": re.compile(
        r"(?i)social security card|social security number"
    ),
    "immigration_document": re.compile(
        r"(?i)form i-20|i-94|employment authorization document|"
        r"permanent resident card"
    ),
}

PROMPT_INJECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_instructions": re.compile(
        r"(?i)ignore (?:all |any )?(?:previous|prior|system) instructions"
    ),
    "system_prompt": re.compile(
        r"(?i)system prompt|developer message|hidden instructions"
    ),
    "tool_request": re.compile(
        r"(?i)call (?:the )?tool|execute (?:this )?command|"
        r"send (?:the )?data"
    ),
    "model_override": re.compile(
        r"(?i)you are now|act as .*assistant|override .*policy"
    ),
}

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.I,
    ),
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)"
        r"\d{3}[-.\s]?\d{4}(?!\d)"
    ),
    "ssn_like": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "credit_card_like": re.compile(
        r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"
    ),
}

SENSITIVE_TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "financial": re.compile(
        r"(?i)bank account|routing number|credit score|loan application|"
        r"tax return|w-2|1042-s"
    ),
    "medical": re.compile(
        r"(?i)diagnosis|medical record|prescription|therapy|mental health|"
        r"health insurance"
    ),
    "legal_or_immigration": re.compile(
        r"(?i)legal notice|court|lawsuit|visa status|immigration record|uscis"
    ),
    "intimate_or_grief": re.compile(
        r"(?i)intimate relationship|bereavement|grief|funeral|died|"
        r"death certificate"
    ),
}


@dataclass(frozen=True)
class PrivacyScanResult:
    secret_types: list[str]
    identity_document_types: list[str]
    prompt_injection_types: list[str]
    pii_counts: dict[str, int]
    sensitive_topics: list[str]
    presidio_counts: dict[str, int]

    @property
    def has_secret(self) -> bool:
        return bool(self.secret_types)

    @property
    def has_identity_document(self) -> bool:
        return bool(self.identity_document_types)

    @property
    def has_prompt_injection(self) -> bool:
        return bool(self.prompt_injection_types)


def _types_found(
    patterns: dict[str, re.Pattern[str]],
    text: str,
) -> list[str]:
    return sorted(
        name for name, pattern in patterns.items() if pattern.search(text)
    )


def _counts(
    patterns: dict[str, re.Pattern[str]],
    text: str,
) -> dict[str, int]:
    return {
        name: min(1000, len(pattern.findall(text)))
        for name, pattern in patterns.items()
        if pattern.search(text)
    }


@lru_cache(maxsize=1)
def _presidio_analyzer():
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError as exc:
        raise RuntimeError(
            "Presidio is not installed. Install requirements-presidio.txt "
            "and a spaCy English model, or omit --use-presidio."
        ) from exc
    return AnalyzerEngine()


def _presidio_counts(text: str) -> dict[str, int]:
    analyzer = _presidio_analyzer()
    results = analyzer.analyze(text=text, language="en")
    counts = Counter(result.entity_type for result in results)
    return dict(counts)


def scan_privacy(
    text: str,
    *,
    metadata_text: str = "",
    use_presidio: bool = False,
) -> PrivacyScanResult:
    combined = f"{metadata_text}\n{text}"
    return PrivacyScanResult(
        secret_types=_types_found(SECRET_PATTERNS, combined),
        identity_document_types=_types_found(
            IDENTITY_PATTERNS,
            combined,
        ),
        prompt_injection_types=_types_found(
            PROMPT_INJECTION_PATTERNS,
            text,
        ),
        pii_counts=_counts(PII_PATTERNS, combined),
        sensitive_topics=_types_found(
            SENSITIVE_TOPIC_PATTERNS,
            combined,
        ),
        presidio_counts=(
            _presidio_counts(text)
            if use_presidio and text.strip()
            else {}
        ),
    )
