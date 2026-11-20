"""Prompt-injection defence for fetched web content.

Pipeline: HTML -> visible text only (hidden elements, comments, attributes and scripts are
dropped but still scanned so they can be reported) -> Unicode normalisation (NFKC, zero-width and
bidi controls removed, common homoglyphs folded) -> sentence-level instruction-pattern detection
(including base64-decoded blobs) -> flagged sentences replaced with a redaction marker -> markup
characters neutralised -> wrapped in a quarantined <untrusted_content> block.

Regex detection is one layer, not a guarantee; see ADR-0002 for the other layers.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser

REDACTED = "[removed: suspected prompt injection]"

ZERO_WIDTH_RE = re.compile("[­͏؜ᅟᅠ឴឵᠎​-‏‪-‮⁠-⁤⁪-⁯﻿]")
_CONFUSABLES = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
        "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "ո": "n", "ν": "v", "ο": "o", "α": "a",
        "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X", "І": "I", "Ѕ": "S",
        "Α": "A", "Ε": "E", "Ο": "O", "Ι": "I", "Τ": "T", "Κ": "K", "Μ": "M", "Ν": "N",
    }
)  # fmt: skip

HIDDEN_TAGS = frozenset({"script", "style", "template", "noscript", "iframe", "object", "svg"})
VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "wbr"}
)
BLOCK_TAGS = frozenset(
    {
        "p", "div", "li", "ul", "ol", "br", "tr", "td", "section", "article", "header",
        "footer", "main", "nav", "aside", "h1", "h2", "h3", "h4", "h5", "h6", "title",
        "blockquote", "pre", "table", "hr", "figure", "figcaption",
    }
)  # fmt: skip
TEXT_ATTRS = ("alt", "title", "content", "aria-label", "placeholder", "value")
_HIDDEN_STYLE = re.compile(
    r"display:none|visibility:hidden|font-size:0(?:px|em|rem|%)?(?:;|$)|opacity:0(?:\.0+)?(?:;|$)"
    r"|(?:left|top|text-indent):-\d{3,}"
)

_RULES: tuple[tuple[str, str], ...] = (
    (
        "override",
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|"
        r"preceding|all|any|your|the)\b.{0,30}\b(instruction|prompt|rule|direction|guideline|"
        r"guardrail)s?\b",
    ),
    (
        "override_multilingual",
        r"\b(ignora|ignorez|ignorer|olvida|ignoriere|vergiss)\b.{0,40}\b(instrucciones|"
        r"instructions|anweisungen|consignes|regeln|reglas)\b",
    ),
    (
        "role_hijack",
        r"\b(you are now|from now on,? you|pretend (to be|you are)|act as (an?|the)|"
        r"developer mode|jailbreak|dan mode)\b",
    ),
    (
        "new_instructions",
        r"\b(new|updated|revised|real|actual|hidden|secret)\s+(system\s+)?(instructions?|"
        r"directives?)\s*:|\bnew system (instructions?|prompt)\b",
    ),
    (
        "addressed_to_ai",
        r"\b(note|message|attention|instructions?|memo)\s+(to|for)\s+(the\s+|any\s+)?(ai|"
        r"assistant|model|llm|agent|chatbot|bot|language model|crawler|scraper)s?\b",
    ),
    (
        "addressed_to_ai_prefix",
        r"^\W*(dear |hey |hi )?(the )?(ai|assistant|ai assistant|language model|llm|agent|"
        r"chatbot|gpt|claude)\b[^:]{0,20}:",
    ),
    ("if_you_are_ai", r"\bif you are an? (ai|language model|llm|assistant|bot|agent)\b"),
    (
        "prompt_exfil",
        r"\b(reveal|print|show|repeat|output|leak|disclose|include|paste)\b.{0,40}\b(system "
        r"prompt|your (instructions|prompt|rules)|hidden prompt|initial prompt)\b",
    ),
    (
        "tool_directive",
        r"\b(call|invoke|run|execute|trigger)\b.{0,30}\b(crm_[a-z_]+|fetch_url|lookup_signals|"
        r"the tool|a tool|function call)\b",
    ),
    (
        "fetch_directive",
        r"\b(must|should|need to|have to|please)\b.{0,25}\b(fetch|visit|open|browse|request|"
        r"load|download)\b.{0,40}(https?://|www\.|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})",
    ),
    (
        "fake_role_tag",
        r"<\s*/?\s*(system|assistant|user|untrusted_content|instructions?|im_start|im_end)\b"
        r"|\[/?(inst|sys|system)\]|<\|im_(start|end)\|>",
    ),
    (
        "copy_directive",
        r"\b(include|insert|add|mention|put|append|embed|place)\b.{0,80}\b(in|into|to|within|"
        r"inside)\s+(the\s+|your\s+|every\s+|each\s+|all\s+)?(email|e-mail|message|copy|"
        r"outreach|reply|response|sequence|draft)s?\b",
    ),
    (
        "copy_link_directive",
        r"\b(make sure|ensure|be sure)\b.{0,40}\b(email|message|copy|outreach|sequence)\b"
        r".{0,40}\b(links?|mentions?|includes?|says?|contains?)\b",
    ),
    (
        "copy_directive_multilingual",
        r"\b(incluye|incluir|añade|agrega|ajoutez|ajouter|füge)\b.{0,60}\b(correo|email|"
        r"mensaje|e-mail|courriel|nachricht)\b",
    ),
    (
        "exfil_address",
        r"\b(send|forward|cc|bcc)\b.{0,40}\b(this|the|all|every|full|complete)\b.{0,30}\b("
        r"conversation|contacts?|list|data|prompt|instructions|leads|crm|records?)\b.{0,40}"
        r"[a-z0-9._%+-]+@[a-z0-9-]+(\.[a-z0-9-]+)+",
    ),
)
# Added after evaluating on real public data (tuned on the datasets' train/validation splits only,
# reported on their held-out test splits; see scripts/eval_real_injection.py and the README).
_RULES_REAL_DATA: tuple[tuple[str, str], ...] = (
    (
        "override_short",
        r"\b(ignore|disregard|forget(\s+about)?|drop|delete|reset|bypass|override|don'?t "
        r"(follow|obey))\s+(all\s+|any\s+|the\s+|your\s+|every\s+|each\s+)?(of\s+)?(your\s+|"
        r"the\s+|my\s+)?(previous\s+|prior\s+|above\s+|preceding\s+|earlier\s+|past\s+|"
        r"original\s+|initial\s+|safety\s+|current\s+)?(instructions?|instrutions|directives?|"
        r"orders|commands|rules|guidelines|restrictions|policies|techniques|text|input|"
        r"statement|prompts?|assignments|tasks|above|previous|prior)\b",
    ),
    (
        "override_everything",
        r"\b(ignore|disregard|forget)\s+(about\s+)?everything\b.{0,40}\b(told|said|before|"
        r"above|so far|up to now|until now|instructions?|discussed|beforehand)\b",
    ),
    (
        "instructions_query",
        r"\b(what|which|tell me|list|repeat|recite|describe|spell|change|reset|delete|invert|"
        r"reverse|ignore|forget|disregard|override|adhere to|obey)\b.{0,30}\b(your|previous|"
        r"prior|initial|original|preserved|the above|the previous|the initial|the original|the "
        r"system|the hidden|all( the)?)\s+(\w+\s+)?(instructions|instrutions|intstructions|"
        r"system prompt|prompt texts?)\b",
    ),
    (
        "instructions_given",
        r"\b(instructions?|instrutions|intstructions|prompt)\b.{0,25}\b(you (were|have been|"
        r"got|received) given|were you given|you('ve| have) been given|given to you|you were "
        r"given|you received)\b",
    ),
    (
        "instructions_changed",
        r"\b(your|the|all|previous)\s+(\w+\s+)?instructions?\s+(have|has|are|were)\s+(all\s+)?"
        r"(now\s+)?(been\s+)?(changed|false|wrong|void|reset|updated|overridden)\b|\bi'?m giving "
        r"you a new instruction\b",
    ),
    (
        "password_request",
        r"\b(reveal|tell me|give me|print|spell|show me|what is|what's|whats|say)\b.{0,25}"
        r"\b(the|your)\s+(secret\s+)?(password|passcode|pass word)\b",
    ),
    (
        "new_task_follows",
        r"\b(now|next)\b.{0,20}\b(new|further|other|more)\s+(tasks?|instructions?|"
        r"assignments?)\s+(follow|are following|come)\b|\b(concentrate|focus)\s+on\s+(your|the)\s+"
        r"new\s+(task|assignment)\b|\bleave all (the )?previous information\b",
    ),
    (
        "prompt_peek",
        r"\bwhat\s+(was|is|were)\s+written\s+(above|before|at the (beginning|start|top))\b|"
        r"\b(show|print|reveal|output|display)\b.{0,30}\bprompt[- ]texts?\b",
    ),
    (
        "act_as_template",
        r"\bi want you to act as\b|\b(stay|remain)s?\s+(fully\s+|always\s+)?in\s+(their|your)\s+"
        r"roles?\b|\bdo not answer (this|the) question\b",
    ),
    (
        "override_german",
        r"\b(ignorier\w*|vergiss|vergessen sie|missacht\w*)\b.{0,50}\b(anweisung\w*|befehl\w*|"
        r"aufgabe\w*|auftr[äa]g\w*|angaben|informationen|vorgab\w*|regeln|alles|obigen|"
        r"vorherig\w*|vorangeh\w*|vorangegangen\w*)\b|\b(obigen|vorherigen|vorangegangenen)\b"
        r".{0,40}\bignorier\w*",
    ),
    (
        "new_task_german",
        r"\b(nun|jetzt)\s+folgen\s+(neue|weitere)\s+(anweisungen|aufgaben)\b|\bkonzentrier\w*\s+"
        r"(dich|sie sich)\s+(jetzt\s+|nun\s+)?auf\s+(deine|ihre|die)\s+neue\s+aufgabe\b|"
        r"\bich möchte, dass (sie|du) als\b.{0,60}\b(fungier\w*|agier\w*)\b|"
        r"\b(nun|jetzt) bist du\b|\bstell dir vor, du bist\b|\bzeig\w*\b.{0,30}\bprompt",
    ),
    ("override_romance", r"\b(olvida|olvide|oublie|oubliez)\s+(todo|tout)\b"),
    (
        "answer_directive",
        r"\b(integrate|incorporate|weave|blend|inject|embed|insert|include|append|add|end|"
        r"conclude|begin|start)\b.{0,80}\b(in|into|to|within|inside|throughout|with)\s+(your|the)"
        r"\s+(\w+\s+)?(answer|response|reply|solution|codebase|implementation)s?\b",
    ),
    (
        "answer_rewrite",
        r"^\W*(modify|augment|alter|adjust|change|render|format|rewrite|encode|translate)\s+your\s+"
        r"(\w+\s+)?(answer|response|reply|message|output)s?\b",
    ),
)

RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _RULES + _RULES_REAL_DATA
)
_B64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Finding:
    rule: str
    location: str
    excerpt: str

    def label(self) -> str:
        return f"{self.rule}@{self.location}"


@dataclass(frozen=True)
class QuarantineResult:
    text: str
    visible_text: str
    findings: tuple[Finding, ...] = field(default_factory=tuple)
    redacted_sentences: int = 0
    hidden_segments: int = 0

    @property
    def flagged(self) -> bool:
        return bool(self.findings)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text)
    t = ZERO_WIDTH_RE.sub("", t)
    t = t.translate(_CONFUSABLES)
    return re.sub(r"[ \t\r\f\v]+", " ", t)


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            out.extend(s.strip() for s in _SENTENCE_SPLIT.split(line) if s.strip())
    return out


def _decode_b64(blob: str) -> str | None:
    padded = blob + "=" * (-len(blob) % 4)
    try:
        raw = base64.b64decode(padded, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(ch.isprintable() or ch.isspace() for ch in decoded)
    return decoded if decoded and printable / len(decoded) > 0.95 else None


Rules = tuple[tuple[str, re.Pattern[str]], ...]


def detect(text: str, location: str = "visible", rules: Rules | None = None) -> list[Finding]:
    """Scan already-normalised text sentence by sentence."""
    active = RULES if rules is None else rules
    findings: list[Finding] = []
    for sentence in split_sentences(text):
        for name, pattern in active:
            if pattern.search(sentence):
                findings.append(Finding(name, location, sentence[:160]))
        for blob in _B64_RE.findall(sentence):
            decoded = _decode_b64(blob)
            if decoded and detect(normalize(decoded), "base64", active):
                findings.append(Finding("encoded_instruction", location, blob[:60]))
    return findings


def _hidden_style(style: str) -> bool:
    return bool(_HIDDEN_STYLE.search(style.replace(" ", "").lower()))


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible: list[str] = []
        self.hidden: list[tuple[str, str]] = []
        self._stack: list[tuple[str, bool]] = []

    def _is_hidden(self) -> bool:
        return any(h for _, h in self._stack)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        for key in TEXT_ATTRS:
            if a.get(key, "").strip():
                self.hidden.append(("attribute", a[key]))
        if tag in BLOCK_TAGS:
            self.visible.append("\n")
        if tag in VOID_TAGS:
            return
        hidden = (
            tag in HIDDEN_TAGS
            or "hidden" in a
            or a.get("aria-hidden", "").lower() == "true"
            or _hidden_style(a.get("style", ""))
        )
        self._stack.append((tag, hidden))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS and self._stack and self._stack[-1][0] == tag:
            self._stack.pop()

    def handle_endtag(self, tag: str) -> None:
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                del self._stack[i:]
                break
        if tag in BLOCK_TAGS:
            self.visible.append("\n")

    def handle_data(self, data: str) -> None:
        if not data.strip():
            self.visible.append(data)
        elif self._is_hidden():
            self.hidden.append(("hidden_element", data))
        else:
            self.visible.append(data)

    def handle_comment(self, data: str) -> None:
        if data.strip():
            self.hidden.append(("comment", data))


def extract(html_text: str) -> tuple[str, list[tuple[str, str]]]:
    parser = _Extractor()
    parser.feed(html_text)
    parser.close()
    return "".join(parser.visible), parser.hidden


def _neutralise_markup(text: str) -> str:
    return text.replace("<", "‹").replace(">", "›")


def quarantine(
    raw: str, source: str, source_id: str, rules: Rules | None = None
) -> QuarantineResult:
    visible_raw, hidden = extract(raw)
    findings: list[Finding] = []
    for location, segment in hidden:
        findings.extend(detect(normalize(segment), location, rules))

    kept: list[str] = []
    redacted = 0
    for sentence in split_sentences(normalize(visible_raw)):
        hits = detect(sentence, "visible", rules)
        if hits:
            findings.extend(hits)
            redacted += 1
            if not kept or kept[-1] != REDACTED:
                kept.append(REDACTED)
        else:
            kept.append(sentence)

    body = _neutralise_markup("\n".join(kept))
    flagged = "true" if findings else "false"
    wrapped = (
        f'<untrusted_content source_id="{escape(source_id)}" source="{escape(source)}" '
        f'flagged="{flagged}">\n{body}\n</untrusted_content>'
    )
    return QuarantineResult(
        text=wrapped,
        visible_text=body,
        findings=tuple(findings),
        redacted_sentences=redacted,
        hidden_segments=len(hidden),
    )
