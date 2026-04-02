"""
Regex-based PII redaction on extracted PDF text before chunking and embedding.

Why: chunk text is embedded (third-party API) and may be sent to an LLM judge.
Replacing emails, phones, national IDs, and labelled names with placeholders
reduces accidental disclosure in prompts, logs, and vector stores—not optional
if those services are outside your trust boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
NATIONAL_ID_RE = re.compile(r"\b[A-Z]{2}\d{6}[A-Z]?\b", re.IGNORECASE)
NAME_FIELD_RE = re.compile(r"\b(Name|Client|Prepared by|Reviewed by)\s*:\s*([A-Za-z][A-Za-z\s'.-]{1,80})")
CONTACT_FIELD_RE = re.compile(
    r"\b(Tel|Telephone|Phone|Mobile|Fax)\s*:\s*(\+?\d[\d\s().-]{6,}\d)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class PiiAuditEntry:
    entity_type: str
    original_value: str
    replacement_value: str


def redact_pii_with_audit(text: str) -> tuple[str, list[PiiAuditEntry]]:
    audits: list[PiiAuditEntry] = []
    counters = {"EMAIL": 0, "PHONE": 0, "ID": 0, "PERSON": 0}

    def _token(kind: str) -> str:
        counters[kind] += 1
        return f"[{kind}_{counters[kind]}]"

    def _replace_regex(pattern: re.Pattern[str], kind: str, source_text: str) -> str:
        def _repl(match: re.Match[str]) -> str:
            replacement = _token(kind)
            audits.append(
                PiiAuditEntry(
                    entity_type=kind.lower(),
                    original_value=match.group(0),
                    replacement_value=replacement,
                )
            )
            return replacement

        return pattern.sub(_repl, source_text)

    output = _replace_regex(EMAIL_RE, "EMAIL", text)
    output = _replace_regex(NATIONAL_ID_RE, "ID", output)

    def _name_repl(match: re.Match[str]) -> str:
        label = match.group(1)
        original = match.group(2).strip()
        replacement = _token("PERSON")
        audits.append(
            PiiAuditEntry(
                entity_type="person",
                original_value=original,
                replacement_value=replacement,
            )
        )
        return f"{label}: {replacement}"

    output = NAME_FIELD_RE.sub(_name_repl, output)

    def _contact_repl(match: re.Match[str]) -> str:
        label = match.group(1)
        original = match.group(2).strip()
        replacement = _token("PHONE")
        audits.append(
            PiiAuditEntry(
                entity_type="phone",
                original_value=original,
                replacement_value=replacement,
            )
        )
        return f"{label}: {replacement}"

    output = CONTACT_FIELD_RE.sub(_contact_repl, output)
    return output, audits


def redact_pii(text: str) -> str:
    redacted, _ = redact_pii_with_audit(text)
    return redacted

