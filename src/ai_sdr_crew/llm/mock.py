"""Deterministic offline LLM.

The mock answers through the same Messages-shaped interface as the live backend, so the whole
pipeline (tool loop, schema validation, allowlist, budgets, caching accounting, revise loop) runs
unchanged offline. Its behaviour is intentionally imperfect and *prompt-sensitive*:

- It follows a rule only if the rule's phrase is present in the system prompt (see RULE_PHRASES),
  so prompt v1 vs v2 produce measurably different first drafts.
- It is GULLIBLE: it reads through obfuscation (comments, hidden text, zero-width characters,
  homoglyphs, base64) and obeys instruction-like text it sees, by calling tools or pasting
  payloads into copy. That makes red-team tests meaningful: only the deterministic defence
  layers stand between injected text and the output.

Token counts are estimates (~4 chars/token) and caching follows Anthropic's rules: the prefix
(tools + system up to the cache breakpoint) is cached if it is >= 1024 tokens.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import itertools
import json
import re
from typing import Any

from ai_sdr_crew.evals.judge import heuristic_rubric
from ai_sdr_crew.guardrails.injection import REDACTED, normalize, split_sentences
from ai_sdr_crew.guardrails.linter import LIMITS, words
from ai_sdr_crew.llm.types import LLMRequest, LLMResponse, TextBlock, ToolResultBlock, ToolUseBlock
from ai_sdr_crew.models import Angle, Sequence, Usage

MIN_CACHEABLE_TOKENS = 1024

RULE_PHRASES = {
    "no_link_step1": "Never put a link in email 1",
    "length": "Keep email 1 under 90 words",
    "no_hype": "Do not use hype",
    "cite_all": "Every factual sentence must end with a citation",
    "opt_out": "End the final email with an opt-out line",
    "no_placeholders": "Never leave placeholders",
    "question_cta": "End email 1 with a single low-friction question",
    "quote_verbatim": "Copy each quote verbatim from a tool output",
    "focus": "Use one signal in email 1",
}
SIGNAL_PRIORITY = {
    "funding": 6, "expansion": 5, "tech_change": 4, "hiring": 4,
    "leadership": 3, "launch": 2, "website": 1, "crm": 0,
}  # fmt: skip
ANGLES = {
    "funding": (
        "A new round means more vendors and more spend landing on a finance team sized for last "
        "year.",
        "Scale AP throughput without adding headcount while the business grows.",
    ),
    "expansion": (
        "New entities and locations multiply approval paths, currencies and vendor records.",
        "One AP workflow with entity-aware approvals instead of a process per location.",
    ),
    "tech_change": (
        "An ERP migration is the cheapest moment to redesign how invoices flow.",
        "Land AP automation alongside the new ERP instead of rebuilding manual steps on it.",
    ),
    "hiring": (
        "Hiring for AP signals invoice volume is outgrowing the current process.",
        "Let the new hire own exceptions and vendors rather than data entry.",
    ),
    "leadership": (
        "A new finance leader is looking for visible quick wins in the first quarter.",
        "A contained AP project that shows results inside a 90-day plan.",
    ),
    "launch": (
        "A new channel or product line brings new vendors and invoice formats.",
        "Absorb new vendor volume without slowing approvals.",
    ),
}
SUBJECTS = {
    "funding": "Scaling AP after the raise",
    "expansion": "AP across new entities",
    "tech_change": "AP during the ERP move",
    "hiring": "Before the new AP hire starts",
    "leadership": "First quarter in finance",
    "launch": "More vendors, same AP team",
}
BRIDGES = {
    "funding": "Growth rounds tend to mean more vendors and invoices for the same finance team.",
    "expansion": "Each new entity tends to add its own approval path and vendor list.",
    "tech_change": "A systems change is usually the best moment to rethink how invoices flow.",
    "hiring": "A new AP hire is a good moment to decide which work should stay manual.",
    "leadership": "A new finance leader usually looks for a few quick, visible wins.",
    "launch": "New channels tend to bring new vendors and new invoice formats.",
}
TOPICS = {
    "funding": "AP after the raise",
    "expansion": "AP across new entities",
    "tech_change": "AP during the ERP move",
    "hiring": "the new AP role",
    "leadership": "early wins in finance",
    "launch": "vendor volume from the launch",
}
PAINS = {
    "funding": "The usual pinch point is approvals: invoices wait on busy budget owners.",
    "expansion": "The usual pinch point is approvals that differ by entity and currency.",
    "tech_change": "The usual pinch point is rebuilding manual approval steps in the new system.",
    "hiring": "The usual pinch point is data entry eating the time meant for vendor issues.",
    "leadership": "The usual pinch point is limited visibility into what is waiting for approval.",
    "launch": "The usual pinch point is onboarding new vendors without slowing payments.",
}
EVENTS = {
    "funding": "the raise",
    "expansion": "the expansion",
    "tech_change": "the ERP move",
    "hiring": "the new hire",
    "leadership": "the leadership change",
    "launch": "the launch",
}
VALUE = (
    "Tallyhaus automates invoice capture, approvals and vendor payments so the team keeps up "
    "without adding headcount."
)
CTA = "Worth a short call next week to compare notes?"
OPT_OUT = "If this isn't relevant, just reply 'no' and I won't follow up."
STEP2_CLOSE = (
    "If useful, I can share the approval workflow template finance teams use most. Open to it?"
)
V1_LINK = "You can see a short demo at https://tallyhaus.example/demo."
V1_EXTRA = (
    "Most teams we work with start with invoice capture, then add approval routing by entity and "
    "amount, then move vendor payments over once the workflow is stable, which keeps the rollout "
    "calm for the controller, the AP team and the approvers across every department in the "
    "business, and it also gives everyone time to get comfortable with the new process."
)
V1_FREE_CTA = "Can I send over a free breakdown of how other teams run AP?"
V1_FLAT_CTA = "Happy to share how similar teams handle it."
V1_STAT = "Tallyhaus customers cut invoice processing time by 80%."
WEBSITE_HINTS = re.compile(
    r"\b(serves?|supplies|operates|founded|locations?|stores|clinics|sites|countries|employees|"
    r"partners|customers|fleet|plants?|offices|students|patients|brands?)\b",
    re.IGNORECASE,
)
INSTRUCTION_CUES = re.compile(
    r"\b(ignore|ignora|disregard|include|incluye|insert|add|mention|call|invoke|fetch|visit|"
    r"send|forward|reveal|print|must|follow|instructions?|links? to)\b|note to (the )?ai|"
    r"ai assistant|assistant:|<\s*system|developer mode",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
CANARY_RE = re.compile(r"CANARY-[A-Z0-9-]+")
MONEY_RE = re.compile(r"\$\d+(?:\.\d+)?\s?[MBK]\b")
B64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}")
SIGNAL_LINE = re.compile(r"^- \[(\w+)\] (\S+) \| (.+?) \| source: (\S+)$")
REMOVE_SENTENCE_CODES = frozenset(
    {
        "LINK_IN_FIRST_EMAIL", "UNCITED_CLAIM", "NUMBER_UNSUPPORTED", "UNKNOWN_CITATION",
        "UNVERIFIABLE_CLAIM", "DISALLOWED_LINK", "EMAIL_ADDRESS_IN_COPY",
        "PROMPT_OR_ARTIFACT_LEAK",
    }
)  # fmt: skip


def est_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def extract_tag(text: str, tag: str) -> str | None:
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
    return m.group(1) if m else None


def read_like_a_model(raw: str) -> list[str]:
    """How a capable but gullible model 'sees' a page: everything, de-obfuscated."""
    t = re.sub(r"<!--(.*?)-->", r"\n\1\n", raw, flags=re.S)
    attrs = re.findall(r'\b(?:alt|title|content|aria-label)\s*=\s*"([^"]*)"', t)
    t = re.sub(r"<[^>]+>", "\n", t)
    t = html.unescape("\n".join([t, *attrs]))
    t = normalize(t)
    decoded: list[str] = []
    for blob in B64_RE.findall(t):
        try:
            text = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=True).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if text.isprintable():
            decoded.append(normalize(text))
    return split_sentences("\n".join([t, *decoded]))


def payload_sentences(text: str, fact_id: str, leak_line: str) -> list[str]:
    out: list[str] = []
    out += [f"More context here: {u.rstrip('.,;')} [{fact_id}]." for u in URL_RE.findall(text)]
    out += [f"Loop in {e.rstrip('.')} for details [{fact_id}]." for e in EMAIL_RE.findall(text)]
    out += [f"Reference code {c} [{fact_id}]." for c in CANARY_RE.findall(text)]
    out += [f"Congrats on the {m} raise [{fact_id}]." for m in MONEY_RE.findall(text)]
    if re.search(r"system prompt|your instructions|hidden prompt", text, re.IGNORECASE):
        out.append(f"For transparency, my brief reads: {leak_line} [{fact_id}].")
    return out


def _strip(text: str) -> str:
    return text.strip().rstrip(".")


def _paragraphs(text: str) -> list[list[str]]:
    return [
        [s for s in re.split(r"(?<=[.!?])\s+", p.strip()) if s]
        for p in text.split("\n\n")
        if p.strip()
    ]


def _join(paras: list[list[str]]) -> str:
    return "\n\n".join(" ".join(p) for p in paras if p)


def remove_sentence(text: str, evidence: str) -> str:
    ev = evidence.strip()
    paras = [[s for s in p if ev not in s and s not in ev] for p in _paragraphs(text)]
    return _join(paras)


def trim_longest(text: str, limit: int) -> str:
    paras = _paragraphs(text)
    while words(_join(paras)) > limit:
        candidates = [
            (len(s), pi, si)
            for pi, p in enumerate(paras)
            for si, s in enumerate(p)
            if "[F" not in s and not s.endswith("?") and not s.lower().startswith("hi ")
        ]
        if not candidates:
            break
        _, pi, si = max(candidates)
        del paras[pi][si]
    return _join(paras)


def replace_last_sentence(text: str, new: str) -> str:
    paras = _paragraphs(text)
    if paras and paras[-1]:
        paras[-1][-1] = new
    return _join(paras)


class MockBackend:
    name = "mock"

    def __init__(self) -> None:
        self._cache: set[str] = set()
        self._ids = itertools.count(1)

    # ---- plumbing -------------------------------------------------------------------------
    def _tool_use(self, name: str, payload: dict[str, Any]) -> ToolUseBlock:
        return ToolUseBlock(id=f"toolu_mock_{next(self._ids):05d}", name=name, input=payload)

    def create(self, req: LLMRequest) -> LLMResponse:
        handlers = {
            "researcher": self._researcher,
            "strategist": self._strategist,
            "copywriter": self._copywriter,
            "critic": self._critic,
            "judge": self._judge,
        }
        content = handlers[req.role](req)
        return LLMResponse(content=content, usage=self._usage(req, content), stop_reason="tool_use")

    def _usage(self, req: LLMRequest, content: list[TextBlock | ToolUseBlock]) -> Usage:
        tools_text = json.dumps([t.model_dump() for t in req.tools], sort_keys=True)
        breakpoints = [i for i, b in enumerate(req.system) if b.cache]
        cache_write = cache_read = 0
        uncached_system = req.system
        prefix_tokens = est_tokens(tools_text)
        if breakpoints:
            idx = breakpoints[-1]
            prefix = tools_text + "".join(b.text for b in req.system[: idx + 1])
            prefix_tokens = est_tokens(prefix)
            uncached_system = req.system[idx + 1 :]
            if prefix_tokens >= MIN_CACHEABLE_TOKENS:
                key = hashlib.sha256(f"{req.model}|{prefix}".encode()).hexdigest()
                if key in self._cache:
                    cache_read = prefix_tokens
                else:
                    cache_write = prefix_tokens
                    self._cache.add(key)
                prefix_tokens = 0
        msg_text = json.dumps([m.model_dump() for m in req.messages])
        input_tokens = prefix_tokens + est_tokens("".join(b.text for b in uncached_system))
        input_tokens += est_tokens(msg_text)
        out_text = json.dumps([b.model_dump() for b in content])
        return Usage(
            input_tokens=input_tokens,
            output_tokens=est_tokens(out_text),
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        )

    @staticmethod
    def _rules(req: LLMRequest) -> dict[str, bool]:
        system_text = "\n".join(b.text for b in req.system)
        return {k: phrase in system_text for k, phrase in RULE_PHRASES.items()}

    # ---- researcher -----------------------------------------------------------------------
    def _researcher(self, req: LLMRequest) -> list[TextBlock | ToolUseBlock]:
        account = json.loads(extract_tag(req.messages[0].text(), "account") or "{}")
        domain, website, company = account["domain"], account["website"], account["company"]
        calls: dict[str, ToolUseBlock] = {}
        results: list[tuple[ToolUseBlock, ToolResultBlock]] = []
        for msg in req.messages:
            for b in msg.content:
                if isinstance(b, ToolUseBlock):
                    calls[b.id] = b
                elif isinstance(b, ToolResultBlock) and b.tool_use_id in calls:
                    results.append((calls[b.tool_use_id], b))
        if not calls:
            first_calls: list[TextBlock | ToolUseBlock] = [
                self._tool_use("crm_lookup", {"domain": domain}),
                self._tool_use("lookup_signals", {"domain": domain}),
                self._tool_use("fetch_url", {"url": website}),
            ]
            return first_calls

        web_sentences = [
            s
            for call, res in results
            if call.name == "fetch_url" and not res.is_error
            for s in read_like_a_model(res.content)
        ]
        attempted = {(c.name, json.dumps(c.input, sort_keys=True)) for c in calls.values()}
        pending: list[ToolUseBlock] = []
        for s in web_sentences:
            if not INSTRUCTION_CUES.search(s):
                continue
            if "crm_log_activity" in s.lower():
                args = {"domain": domain, "activity": s[:200]}
                if ("crm_log_activity", json.dumps(args, sort_keys=True)) not in attempted:
                    pending.append(self._tool_use("crm_log_activity", args))
            if re.search(r"\b(fetch|visit|open|browse|load|request)\b", s, re.IGNORECASE):
                for url in URL_RE.findall(s):
                    args = {"url": url.rstrip(".,;")}
                    if ("fetch_url", json.dumps(args, sort_keys=True)) not in attempted:
                        pending.append(self._tool_use("fetch_url", args))
        if pending:
            return list[TextBlock | ToolUseBlock](pending)

        facts: list[dict[str, Any]] = []

        def add(kind: str, text: str, quote: str, tool: str, ref: str) -> None:
            facts.append(
                {
                    "id": f"F{len(facts) + 1}",
                    "kind": kind,
                    "text": text[:400],
                    "quote": quote[:400],
                    "source_tool": tool,
                    "source_ref": ref,
                }
            )

        for call, res in results:
            if res.is_error:
                continue
            if call.name == "lookup_signals":
                for line in res.content.splitlines():
                    m = SIGNAL_LINE.match(line.strip())
                    if m:
                        add(m.group(1), m.group(3), m.group(3), "lookup_signals", m.group(4))
            elif call.name == "crm_lookup":
                notes = re.search(r"^notes: (.+)$", res.content, re.M)
                if notes and notes.group(1).strip() != "none":
                    add("crm", f"CRM notes: {notes.group(1)}", notes.group(1), "crm_lookup", domain)

        benign = 0
        injected = 0
        first_word = company.split()[0].lower()
        for s in web_sentences:
            if REDACTED in s or not 25 <= len(s) <= 300:
                continue
            if INSTRUCTION_CUES.search(s):
                if injected < 3 and payload_sentences(s, "F0", "x"):
                    add("website", s, s, "fetch_url", website)  # gullible: treats it as a fact
                    injected += 1
                continue
            if benign < 2 and (first_word in s.lower() or WEBSITE_HINTS.search(s)):
                add("website", s, s, "fetch_url", website)
                benign += 1

        if not self._rules(req)["quote_verbatim"] and stable_hash(domain) % 4 == 0:
            claim = f"{company} is preparing for an IPO in 2027"
            add("funding", claim + ".", claim, "fetch_url", website)  # hallucination

        summary = f"{len(facts)} facts collected for {company}."
        return [self._tool_use("submit_research_brief", {"facts": facts, "summary": summary})]

    # ---- strategist -----------------------------------------------------------------------
    def _strategist(self, req: LLMRequest) -> list[TextBlock | ToolUseBlock]:
        facts: list[dict[str, Any]] = json.loads(
            extract_tag(req.messages[0].text(), "facts") or "[]"
        )
        ranked = sorted(
            enumerate(facts), key=lambda p: (-SIGNAL_PRIORITY.get(p[1]["kind"], 0), p[0])
        )
        best = ranked[0][1]
        kind = best["kind"] if best["kind"] in ANGLES else "hiring"
        hypothesis, value_prop = ANGLES[kind]
        angle = {
            "signal_fact_id": best["id"],
            "kind": best["kind"],
            "hypothesis": hypothesis,
            "value_prop": value_prop,
        }
        return [self._tool_use("submit_angle", angle)]

    # ---- copywriter -----------------------------------------------------------------------
    def _copywriter(self, req: LLMRequest) -> list[TextBlock | ToolUseBlock]:
        text = req.messages[0].text()
        account = json.loads(extract_tag(text, "account") or "{}")
        facts: list[dict[str, Any]] = json.loads(extract_tag(text, "facts") or "[]")
        angle = json.loads(extract_tag(text, "angle") or "{}")
        draft = extract_tag(text, "draft")
        if draft:
            issues = json.loads(extract_tag(text, "issues") or "[]")
            seq = self._revise(json.loads(draft), issues, account)
        else:
            leak_line = req.system[0].text.strip().splitlines()[0].lstrip("# ").strip()
            seq = self._compose(account, facts, angle, self._rules(req), leak_line)
        return [self._tool_use("submit_sequence", seq)]

    def _compose(
        self,
        account: dict[str, Any],
        facts: list[dict[str, Any]],
        angle: dict[str, Any],
        rules: dict[str, bool],
        leak_line: str,
    ) -> dict[str, Any]:
        h = stable_hash(account["domain"])
        first = account["contact"].get("first_name")
        name = first or ("there" if rules["no_placeholders"] else "{{first_name}}")
        by_id = {f["id"]: f for f in facts}
        af = by_id.get(angle.get("signal_fact_id", ""), facts[0])
        kind = str(angle.get("kind")) if angle.get("kind") in SUBJECTS else "hiring"
        injected = [
            p
            for f in facts
            if INSTRUCTION_CUES.search(f["text"])
            for p in payload_sentences(f["text"], f["id"], leak_line)
        ]
        secondary = next(
            (
                f
                for f in facts
                if f["kind"] == "website"
                and f["id"] != af["id"]
                and not INSTRUCTION_CUES.search(f["text"])
            ),
            None,
        )
        site_line = (
            f"I also noticed this on your site: {_strip(secondary['text'])} [{secondary['id']}]."
            if secondary
            else ""
        )

        p2 = f"Saw the news: {_strip(af['text'])} [{af['id']}]. {BRIDGES[kind]}"
        if site_line and not rules["focus"]:
            p2 += " " + site_line
        if not rules["no_link_step1"] and h % 3 == 0:
            p2 += " " + V1_LINK
        cta = CTA
        if not rules["no_hype"] and h % 4 == 1:
            cta = V1_FREE_CTA
        elif not rules["question_cta"] and h % 5 == 0:
            cta = V1_FLAT_CTA
        paras = [f"Hi {name},", p2, *([" ".join(injected)] if injected else [])]
        if not rules["length"] and h % 3 == 1:
            paras.append(V1_EXTRA)
        paras.append(f"{VALUE} {cta}")
        step1 = "\n\n".join(paras)

        p_mid = f"Following up on my note about {TOPICS[kind]}. {PAINS[kind]}"
        if site_line and rules["focus"]:
            p_mid += " " + site_line
        if not rules["cite_all"] and h % 2 == 0:
            p_mid += " " + V1_STAT
        step2 = "\n\n".join([f"Hi {name},", p_mid, STEP2_CLOSE])

        close = (
            f"I'll close the loop here. If AP automation becomes a priority after {EVENTS[kind]}, "
            "I'm happy to help."
        )
        step3_paras = [f"Hi {name},", close]
        if rules["opt_out"] or h % 2 == 1:
            step3_paras.append(OPT_OUT)
        note = (
            f"Hi {name}, saw the news about {account['company']} [{af['id']}]. I work with finance "
            "teams on AP automation and would be glad to connect."
        )
        return {
            "steps": [
                {"step": 1, "delay_days": 0, "subject": SUBJECTS[kind], "body": step1},
                {"step": 2, "delay_days": 3, "subject": "One more thought on AP", "body": step2},
                {
                    "step": 3,
                    "delay_days": 5,
                    "subject": "Closing the loop",
                    "body": "\n\n".join(step3_paras),
                },
            ],
            "linkedin_note": note,
        }

    def _revise(
        self, draft: dict[str, Any], issues: list[dict[str, Any]], account: dict[str, Any]
    ) -> dict[str, Any]:
        first = account["contact"].get("first_name") or "there"
        texts: dict[str, str] = {}
        for s in draft["steps"]:
            texts[f"subject{s['step']}"] = s["subject"]
            texts[f"step{s['step']}"] = s["body"]
        texts["linkedin"] = draft["linkedin_note"]
        for issue in issues:
            code, loc, ev = issue["code"], issue.get("location"), issue.get("evidence")
            targets = [loc] if loc in texts else list(texts)
            if code in REMOVE_SENTENCE_CODES and ev:
                for t in targets:
                    texts[t] = remove_sentence(texts[t], ev)
            elif code == "SPAM_WORD" and ev:
                for t in targets:
                    pattern = rf"(?<![\w-]){re.escape(ev)}(?![\w-])\s*"
                    texts[t] = re.sub(pattern, "", texts[t], flags=re.IGNORECASE)
            elif code == "UNFILLED_TOKEN" and ev:
                for t in targets:
                    texts[t] = texts[t].replace(ev, first)
            elif code == "TOO_LONG" and loc in texts:
                limit = LIMITS["step1_words"] if loc == "step1" else LIMITS["followup_words"]
                texts[loc] = trim_longest(texts[loc], limit)
            elif code == "MISSING_OPT_OUT":
                texts["step3"] = texts["step3"].rstrip() + "\n\n" + OPT_OUT
            elif code == "CRITIC_CTA":
                texts["step1"] = replace_last_sentence(texts["step1"], CTA)
        return {
            "steps": [
                {
                    "step": s["step"],
                    "delay_days": s["delay_days"],
                    "subject": texts[f"subject{s['step']}"] or "Quick question",
                    "body": texts[f"step{s['step']}"],
                }
                for s in draft["steps"]
            ],
            "linkedin_note": texts["linkedin"],
        }

    # ---- critic / judge -------------------------------------------------------------------
    def _parse_review_inputs(self, req: LLMRequest) -> tuple[Sequence, Angle]:
        text = req.messages[0].text()
        seq = Sequence.model_validate_json(extract_tag(text, "draft") or "{}")
        angle = Angle.model_validate_json(extract_tag(text, "angle") or "{}")
        return seq, angle

    def _critic(self, req: LLMRequest) -> list[TextBlock | ToolUseBlock]:
        seq, angle = self._parse_review_inputs(req)
        scores, issues = heuristic_rubric(seq, angle)
        verdict = "pass" if scores.minimum() >= 3 else "revise"
        payload = {"scores": scores.model_dump(), "issues": issues, "verdict": verdict}
        return [self._tool_use("submit_critic_review", payload)]

    def _judge(self, req: LLMRequest) -> list[TextBlock | ToolUseBlock]:
        seq, angle = self._parse_review_inputs(req)
        scores, _ = heuristic_rubric(seq, angle)
        return [self._tool_use("submit_rubric_scores", scores.model_dump())]
