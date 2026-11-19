"""Output guard: last line of defence on generated copy before a human ever sees it.

Blocks links to domains other than the seller's and the prospect's, any email address, prompt
leakage markers and quarantine artefacts.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from ai_sdr_crew.guardrails.citations import sentences
from ai_sdr_crew.models import Issue, Sequence

URL_RE = re.compile(r"(?:https?://|www\.)[^\s)\]>\"']+", re.IGNORECASE)
BARE_DOMAIN_RE = re.compile(
    r"(?<![@\w.-])((?:[a-z0-9-]+\.)+(?:com|io|net|org|ai|co|example|xyz|ru|info|biz))\b",
    re.IGNORECASE,
)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
LEAK_MARKERS = (
    "th-shared-prompt",
    "## security policy",
    "## evidence protocol",
    "untrusted_content",
    "[removed: suspected prompt injection]",
    "akiafakemetadatacanary",
)


def _host(url: str) -> str:
    url = url.rstrip(".,;:!?")
    if not url.lower().startswith("http"):
        url = "http://" + url
    return (urlsplit(url).hostname or "").lower().rstrip(".")


def _allowed(host: str, allowed_domains: tuple[str, ...]) -> bool:
    host = host.removeprefix("www.")
    return any(host == d or host.endswith("." + d) for d in allowed_domains)


def guard_sequence(seq: Sequence, allowed_domains: tuple[str, ...]) -> list[Issue]:
    allowed = tuple(d.lower().removeprefix("www.") for d in allowed_domains)
    issues: list[Issue] = []
    for loc, text in seq.texts():
        for sentence in sentences(text):
            lowered = sentence.lower()
            hosts = [_host(u) for u in URL_RE.findall(sentence)]
            hosts += [m.lower() for m in BARE_DOMAIN_RE.findall(sentence)]
            for host in sorted(set(hosts)):
                if host and not _allowed(host, allowed):
                    issues.append(
                        Issue(
                            source="output_guard",
                            code="DISALLOWED_LINK",
                            message=f"link to non-allowlisted domain {host}",
                            location=loc,
                            evidence=sentence,
                        )
                    )
            for email in EMAIL_RE.findall(sentence):
                issues.append(
                    Issue(
                        source="output_guard",
                        code="EMAIL_ADDRESS_IN_COPY",
                        message=f"email address {email} in copy",
                        location=loc,
                        evidence=sentence,
                    )
                )
            for marker in LEAK_MARKERS:
                if marker in lowered:
                    issues.append(
                        Issue(
                            source="output_guard",
                            code="PROMPT_OR_ARTIFACT_LEAK",
                            message=f"internal marker '{marker}' in copy",
                            location=loc,
                            evidence=sentence,
                        )
                    )
    return issues
