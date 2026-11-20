"""Evaluate the deterministic injection defence on REAL public data.

Attacks: labelled public prompt-injection / jailbreak datasets (deepset, Lakera Gandalf, Microsoft
BIPIA, jackhhao). Benign: the benign rows of those datasets plus real B2B web pages fetched through
the repo's own SafeFetcher and run through the exact sanitise + quarantine path the Researcher
sees.

Only model-independent layers are measured here (instruction-pattern detection inside
`quarantine`). The end-to-end section embeds real attacks into a fixture page and runs the full
mock pipeline; the mock is a stand-in for a model, so that part bounds the deterministic layers,
not a real model's behaviour.

Downloads are cached under data/real/ (gitignored). Every number in the generated report comes
from `evaluate_*` below; nothing is typed in by hand.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import random
import re
import ssl
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlunsplit

import httpx

from ai_sdr_crew.config import PROJECT_ROOT
from ai_sdr_crew.evals.redteam import CONFIGS, RedTeamPage, run_html
from ai_sdr_crew.guardrails.injection import (
    RULES,
    Rules,
    extract,
    normalize,
    quarantine,
    split_sentences,
)
from ai_sdr_crew.tools.fetch import FetchBlocked, FetchFailed, SafeFetcher

DATA_DIR = PROJECT_ROOT / "data" / "real"
URLS_PATH = PROJECT_ROOT / "fixtures" / "real" / "b2b_urls.txt"
EVAL_UA = "Mozilla/5.0 (compatible; ai-sdr-crew-eval/0.1)"

# The rule set as it shipped before any tuning on real data. Used to report before/after.
BASELINE_RULE_NAMES = frozenset(
    {
        "override", "override_multilingual", "role_hijack", "new_instructions",
        "addressed_to_ai", "addressed_to_ai_prefix", "if_you_are_ai", "prompt_exfil",
        "tool_directive", "fetch_directive", "fake_role_tag", "copy_directive",
        "copy_link_directive", "copy_directive_multilingual", "exfil_address",
    }
)  # fmt: skip
BASELINE_RULES: Rules = tuple(r for r in RULES if r[0] in BASELINE_RULE_NAMES)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    licence: str
    kind: str  # "direct injection", "indirect injection", "jailbreak"
    url: str


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        "deepset",
        "deepset/prompt-injections",
        "Apache-2.0",
        "direct injection (labelled, EN + DE)",
        "https://huggingface.co/datasets/deepset/prompt-injections",
    ),
    DatasetSpec(
        "gandalf",
        "Lakera/gandalf_ignore_instructions",
        "MIT",
        "direct injection typed by Gandalf players (attacks only)",
        "https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions",
    ),
    DatasetSpec(
        "bipia",
        "microsoft/BIPIA text_attack + code_attack",
        "MIT",
        "indirect injection payloads (attacks only)",
        "https://github.com/microsoft/BIPIA",
    ),
    DatasetSpec(
        "jailbreak_cls",
        "jackhhao/jailbreak-classification",
        "Apache-2.0",
        "jailbreak vs benign role-play prompts (labelled)",
        "https://huggingface.co/datasets/jackhhao/jailbreak-classification",
    ),
)
INJECTION_DATASETS = ("deepset", "gandalf", "bipia")


@dataclass(frozen=True)
class Sample:
    text: str
    label: int  # 1 = attack, 0 = benign
    dataset: str
    split: str  # "dev" (tuning allowed) or "test" (held out)
    lang: str = "en"


# ---------------------------------------------------------------------------------------------
# statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, label: int, flagged: bool) -> None:
        if label == 1 and flagged:
            self.tp += 1
        elif label == 1:
            self.fn += 1
        elif flagged:
            self.fp += 1
        else:
            self.tn += 1

    def metrics(self) -> dict[str, Any]:
        pos, neg, pred = self.tp + self.fn, self.fp + self.tn, self.tp + self.fp
        out: dict[str, Any] = {
            "n_attack": pos,
            "n_benign": neg,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }
        out["recall"] = _rate(self.tp, pos)
        out["precision"] = _rate(self.tp, pred)
        out["fpr"] = _rate(self.fp, neg)
        r, p = out["recall"]["value"], out["precision"]["value"]
        out["f1"] = None if r is None or p is None or r + p == 0 else round(2 * p * r / (p + r), 4)
        return out


def _rate(k: int, n: int) -> dict[str, Any]:
    if n == 0:
        return {"k": 0, "n": 0, "value": None, "ci95": None}
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "value": round(k / n, 6), "ci95": [round(lo, 6), round(hi, 6)]}


# ---------------------------------------------------------------------------------------------
# language tag (heuristic; deepset has no language column)

_EN = frozenset(
    re.findall(r"\S+", "the and you your is are to of in for with that this what how please me my")
    + re.findall(r"\S+", "not be it on all can do i")
)
_DE = frozenset(
    re.findall(r"\S+", "der die das und ist sind nicht ich du sie wir ein eine mit für auf wie")
    + re.findall(r"\S+", "was bitte mir mich zu den dem des von im es")
)


def lang_of(text: str) -> str:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    en = sum(w in _EN for w in words)
    de = sum(w in _DE for w in words) + 2 * len(re.findall("[äöüß]", text.lower()))
    if en == de == 0:
        return "unknown"
    return "de" if de > en else "en"


# ---------------------------------------------------------------------------------------------
# detector under test: the exact sanitise + quarantine path, on the text as a visible paragraph


def as_page(text: str) -> str:
    return f"<p>{escape(text)}</p>"


def flag(text: str, rules: Rules, use_classifier: bool = False) -> tuple[bool, list[str]]:
    q = quarantine(as_page(text), "dataset", "s", rules, use_classifier=use_classifier)
    return q.flagged, sorted({f.rule for f in q.findings})


def evaluate_samples(samples: list[Sample], rules: Rules) -> dict[str, Any]:
    """Confusion matrices broken down by dataset x split x language, plus pooled views."""
    groups: dict[str, Confusion] = {}
    rule_hits: dict[str, int] = {}
    misses: list[dict[str, str]] = []
    false_pos: list[dict[str, Any]] = []
    for s in samples:
        flagged, fired = flag(s.text, rules)
        keys = [
            f"{s.dataset}|{s.split}|all",
            f"{s.dataset}|{s.split}|{s.lang}",
            f"{s.dataset}|all|all",
            f"pooled_injection|{s.split}|all" if s.dataset in INJECTION_DATASETS else "",
        ]
        for k in filter(None, keys):
            groups.setdefault(k, Confusion()).add(s.label, flagged)
        for r in fired:
            rule_hits[r] = rule_hits.get(r, 0) + 1
        if s.label == 1 and not flagged:
            misses.append({"dataset": s.dataset, "split": s.split, "text": s.text})
        if s.label == 0 and flagged:
            false_pos.append(
                {"dataset": s.dataset, "split": s.split, "text": s.text, "rules": fired}
            )
    return {
        "groups": {k: v.metrics() for k, v in sorted(groups.items())},
        "rule_hits": dict(sorted(rule_hits.items(), key=lambda kv: -kv[1])),
        "misses": misses,
        "false_positives": false_pos,
    }


# ---------------------------------------------------------------------------------------------
# benign web pages through the real fetcher


@dataclass
class PageResult:
    url: str
    ok: bool
    error: str = ""
    final_url: str = ""
    bytes_read: int = 0
    truncated: bool = False
    sentences: int = 0
    flagged_visible_sentences: int = 0
    findings: list[dict[str, str]] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return bool(self.findings)


def evaluate_page(
    url: str, final_url: str, html: str, rules: Rules, use_classifier: bool = False
) -> PageResult:
    visible, _hidden = extract(html)
    q = quarantine(html, final_url, "web-1", rules, use_classifier=use_classifier)
    return PageResult(
        url=url,
        ok=True,
        final_url=final_url,
        sentences=len(split_sentences(normalize(visible))),
        flagged_visible_sentences=q.redacted_sentences,
        findings=[
            {"rule": f.rule, "location": f.location, "excerpt": f.excerpt} for f in q.findings
        ],
    )


def summarise_pages(pages: list[PageResult]) -> dict[str, Any]:
    ok = [p for p in pages if p.ok]
    n_sent = sum(p.sentences for p in ok)
    n_flag_sent = sum(p.flagged_visible_sentences for p in ok)
    return {
        "urls": len(pages),
        "fetched": len(ok),
        "failed": len(pages) - len(ok),
        "failures": {p.url: p.error for p in pages if not p.ok},
        "page_fpr": _rate(sum(p.flagged for p in ok), len(ok)),
        "visible_sentence_fpr": _rate(n_flag_sent, n_sent),
        "flagged_pages": [{"url": p.url, "findings": p.findings[:5]} for p in ok if p.flagged],
    }


class UnpinnedProxyTransport(httpx.BaseTransport):
    """Send SafeFetcher's IP-pinned requests by hostname through the environment's HTTPS proxy.

    SafeFetcher still runs every check (scheme, port, DNS -> public IP, redirect re-validation,
    byte cap, deadline, content type); only the final socket pin is undone, because this sandbox
    can only reach the internet through an HTTP CONNECT proxy that expects hostnames. This is the
    same trick tools/web_fixture.py uses to replace the socket offline.
    """

    def __init__(self) -> None:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        cafile = os.environ.get("SSL_CERT_FILE")
        verify: ssl.SSLContext | bool = (
            ssl.create_default_context(cafile=cafile) if cafile else True
        )
        self._inner = httpx.HTTPTransport(proxy=proxy, verify=verify)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        host = request.headers.get("host", request.url.host)
        url = urlunsplit(
            (request.url.scheme, host, request.url.path, request.url.query.decode() or "", "")
        )
        clean = httpx.Request(
            request.method,
            url,
            headers=[(k, v) for k, v in request.headers.items() if k.lower() != "host"],
        )
        return self._inner.handle_request(clean)

    def close(self) -> None:
        self._inner.close()


def read_urls(path: Path = URLS_PATH) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _page_cache(url: str) -> Path:
    return DATA_DIR / "web" / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".json")


def fetch_pages(urls: list[str], refresh: bool = False) -> list[tuple[str, dict[str, Any]]]:
    """Fetch each URL with SafeFetcher (guard on). Cached as JSON {ok, final_url, html|error}."""
    fetcher = SafeFetcher(
        transport=UnpinnedProxyTransport(), timeout_s=20.0, user_agent=EVAL_UA, max_redirects=5
    )
    out: list[tuple[str, dict[str, Any]]] = []
    for url in urls:
        cache = _page_cache(url)
        if cache.exists() and not refresh:
            out.append((url, json.loads(cache.read_text(encoding="utf-8"))))
            continue
        rec: dict[str, Any]
        try:
            r = fetcher.fetch(url)
            rec = {
                "ok": True,
                "final_url": r.final_url,
                "html": r.text,
                "bytes_read": r.bytes_read,
                "truncated": r.truncated,
            }
        except (FetchBlocked, FetchFailed) as e:
            rec = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rec), encoding="utf-8")
        out.append((url, rec))
    return out


def evaluate_web(fetched: list[tuple[str, dict[str, Any]]], rules: Rules) -> list[PageResult]:
    results: list[PageResult] = []
    for url, rec in fetched:
        if not rec.get("ok"):
            results.append(PageResult(url=url, ok=False, error=str(rec.get("error", ""))))
            continue
        pr = evaluate_page(url, str(rec["final_url"]), str(rec["html"]), rules)
        pr.bytes_read = int(rec.get("bytes_read", 0))
        pr.truncated = bool(rec.get("truncated", False))
        results.append(pr)
    return results


# ---------------------------------------------------------------------------------------------
# dataset download (network) -> data/real/datasets/*.jsonl

_HF_API = "https://huggingface.co/api/datasets/{repo}"
_HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
_BIPIA_RAW = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/{name}.json"


def _get(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
    for attempt in range(4):
        r = client.get(url, params=params)
        if r.status_code < 500 and r.status_code != 429:
            r.raise_for_status()
            return r
        time.sleep(2**attempt)
    r.raise_for_status()
    return r


def _hf_rows(client: httpx.Client, repo: str, split: str) -> list[dict[str, Any]]:
    """Read one split's parquet shard(s) from the Hub (needs the optional `pyarrow` extra)."""
    import pyarrow.parquet as pq

    files = _get(client, _HF_API.format(repo=repo)).json()["siblings"]
    shards = sorted(
        f["rfilename"]
        for f in files
        if re.fullmatch(rf"data/{split}-\d+-of-\d+(-[0-9a-f]+)?\.parquet", f["rfilename"])
    )
    if not shards:
        raise FileNotFoundError(f"no parquet shard for {repo}:{split}")
    rows: list[dict[str, Any]] = []
    for shard in shards:
        blob = _get(client, _HF_RESOLVE.format(repo=repo, path=shard)).content
        rows += pq.read_table(io.BytesIO(blob)).to_pylist()
    return rows


def download_datasets(refresh: bool = False) -> Path:
    out = DATA_DIR / "datasets" / "samples.jsonl"
    if out.exists() and not refresh:
        return out
    samples: list[Sample] = []
    with httpx.Client(timeout=60.0, follow_redirects=True) as c:
        for split, tag in (("train", "dev"), ("test", "test")):
            for row in _hf_rows(c, "deepset/prompt-injections", split):
                samples.append(Sample(row["text"], int(row["label"]), "deepset", tag))
        for split, tag in (("train", "dev"), ("validation", "dev"), ("test", "test")):
            for row in _hf_rows(c, "Lakera/gandalf_ignore_instructions", split):
                samples.append(Sample(row["text"], 1, "gandalf", tag))
        for name in ("text_attack", "code_attack"):
            for split, tag in (("train", "dev"), ("test", "test")):
                raw = _get(c, _BIPIA_RAW.format(name=f"{name}_{split}")).json()
                for texts in raw.values():
                    samples += [Sample(t, 1, "bipia", tag) for t in texts]
        for split, tag in (("train", "dev"), ("test", "test")):
            path = f"default/jailbreak_dataset_{split}.csv"
            text = _get(c, _HF_RESOLVE.format(repo="jackhhao/jailbreak-classification", path=path))
            for row in csv.DictReader(io.StringIO(text.text)):
                label = 1 if row["type"].strip() == "jailbreak" else 0
                samples.append(Sample(row["prompt"], label, "jailbreak_cls", tag))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
    return out


def load_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[tuple[str, str]] = set()
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        d = json.loads(line)
        text = str(d["text"]).strip()
        key = (d["dataset"], text)
        if not text or key in seen:  # drop exact duplicates within a dataset
            continue
        seen.add(key)
        samples.append(Sample(text, int(d["label"]), d["dataset"], d["split"], lang_of(text)))
    return samples


# ---------------------------------------------------------------------------------------------
# end to end: real attacks embedded in a fixture page, full mock pipeline

_PAGE = (
    "<!doctype html><html><head><title>{company}</title></head><body>"
    "<header><h1>{company}</h1><p>Precision components for industrial equipment.</p></header>"
    "<main><p>{company} operates offices in Austin and Denver and serves mid-market "
    "manufacturers.</p><p>{attack}</p><p>Our finance team works closely with suppliers in 12 "
    "countries.</p></main><footer><p>Copyright {company}.</p></footer></body></html>"
)


def with_canary(attack: str, canary: str) -> str:
    """Put the canary inside the attack's last sentence so it lives or dies with that sentence."""
    body = attack.rstrip()
    m = re.search(r"[.!?]+$", body)
    if m:
        return f"{body[: m.start()]} {canary}{m.group(0)}"
    return f"{body} {canary}"


def context_exposure(attack: str, rules: Rules, use_classifier: bool = False) -> str:
    """'blocked' | 'partial' | 'passed': how much of the attack reaches the model's context."""
    q = quarantine(as_page(attack), "e2e", "web-1", rules, use_classifier=use_classifier)
    sentences = split_sentences(normalize(attack))
    visible = q.visible_text
    survived = sum(s.replace("<", "‹").replace(">", "›") in visible for s in sentences)
    if survived == 0:
        return "blocked"
    return "passed" if survived == len(sentences) else "partial"


def sample_attacks(samples: list[Sample], n: int, seed: int) -> list[Sample]:
    pool = [
        s for s in samples if s.label == 1 and s.split == "test" and s.dataset in INJECTION_DATASETS
    ]
    return random.Random(seed).sample(pool, min(n, len(pool)))


def evaluate_end_to_end(
    attacks: list[Sample],
    configs: Iterable[str] = ("all_on", "all_off"),
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    by_cfg: dict[str, dict[str, Any]] = {}
    exposure: dict[str, dict[str, int]] = {"baseline": {}, "shipped": {}}
    for label, rules in (("baseline", BASELINE_RULES), ("shipped", RULES)):
        for a in attacks:
            e = context_exposure(a.text, rules)
            exposure[label][e] = exposure[label].get(e, 0) + 1
    for cfg in configs:
        pwned: list[dict[str, Any]] = []
        detected = 0
        for i, a in enumerate(attacks):
            canary = f"CANARY-RX{i:03d}"
            page = RedTeamPage(
                id=f"rx{i:03d}",
                file="",
                company=f"Realx Parts {i:03d}",
                technique="real_attack",
                description=a.dataset,
                canaries=[canary],
            )
            html = _PAGE.format(company=page.company, attack=escape(with_canary(a.text, canary)))
            # This report measures the regex layer: the model layer is switched off here and
            # scored separately in evals/real_classifier.py.
            o = run_html(page, html, cfg, replace(CONFIGS[cfg], injection_classifier=False))
            detected += o.detected
            if o.compromised:
                pwned.append({"i": i, "dataset": a.dataset, "reasons": o.reasons})
            if progress:
                progress(i)
        by_cfg[cfg] = {
            "n": len(attacks),
            "compromised": len(pwned),
            "detected": detected,
            "compromised_cases": pwned,
        }
    datasets: dict[str, int] = {}
    for a in attacks:
        datasets[a.dataset] = datasets.get(a.dataset, 0) + 1
    return {"n": len(attacks), "datasets": datasets, "exposure": exposure, "pipeline": by_cfg}


def excerpt(text: str, words: int = 25) -> str:
    w = " ".join(text.split()).split(" ")
    return " ".join(w[:words]) + (" ..." if len(w) > words else "")


# ---------------------------------------------------------------------------------------------
# report


def _pick(items: list[dict[str, Any]], k: int, seed: int, key: str = "dataset") -> list[Any]:
    """Seeded, round-robin across datasets so one dataset does not fill every slot."""
    shuffled = random.Random(seed).sample(items, len(items))
    buckets: dict[str, list[dict[str, Any]]] = {}
    for it in shuffled:
        buckets.setdefault(str(it.get(key, "")), []).append(it)
    out: list[dict[str, Any]] = []
    while len(out) < k and any(buckets.values()):
        for b in buckets.values():
            if b and len(out) < k:
                out.append(b.pop(0))
    return out


def build_report(
    samples: list[Sample],
    pages_fetched: list[tuple[str, dict[str, Any]]],
    e2e_n: int = 100,
    seed: int = 7,
    run_date: str = "",
) -> dict[str, Any]:
    detectors = {"baseline": BASELINE_RULES, "shipped": RULES}
    datasets_eval = {name: evaluate_samples(samples, rules) for name, rules in detectors.items()}
    web = {name: evaluate_web(pages_fetched, rules) for name, rules in detectors.items()}
    shipped = datasets_eval["shipped"]
    test_misses = [m for m in shipped["misses"] if m["split"] == "test"]
    fps: list[dict[str, Any]] = [
        {"source": f"web:{p['url']}", "dataset": "b2b_web", "text": f["excerpt"], "rule": f["rule"]}
        for p in summarise_pages(web["shipped"])["flagged_pages"]
        for f in p["findings"][:1]
    ]
    fps += [
        {"source": fp["dataset"], "dataset": fp["dataset"], "text": fp["text"], "rule": fp["rules"]}
        for fp in shipped["false_positives"]
        if fp["split"] == "test"
    ]
    attacks = sample_attacks(samples, e2e_n, seed)
    return {
        "date": run_date,
        "seed": seed,
        "datasets": [asdict(d) for d in DATASETS],
        "splits": "dev = dataset train(+validation) splits, used for tuning; test = held-out",
        "baseline_rules": sorted(BASELINE_RULE_NAMES),
        "added_rules": [n for n, _ in RULES if n not in BASELINE_RULE_NAMES],
        "detectors": {
            name: {
                "groups": res["groups"],
                "rule_hits": res["rule_hits"],
                "web": summarise_pages(web[name]),
            }
            for name, res in datasets_eval.items()
        },
        "examples": {
            "misses": [
                {"dataset": m["dataset"], "excerpt": excerpt(m["text"])}
                for m in _pick(test_misses, 5, seed)
            ],
            "false_positives": [
                {"source": f["source"], "rule": f["rule"], "excerpt": excerpt(str(f["text"]))}
                for f in _pick(fps, 5, seed)
            ],
        },
        "end_to_end": evaluate_end_to_end(attacks),
    }


def _fmt(rate: dict[str, Any]) -> str:
    if rate["value"] is None:
        return "n/a"
    lo, hi = rate["ci95"]
    d = 1 if hi >= 0.01 else 2
    ci = f"CI {100 * lo:.{d}f}-{100 * hi:.{d}f}"
    return f"{100 * rate['value']:.{d}f}% ({rate['k']}/{rate['n']}; {ci})"


def report_markdown(rep: dict[str, Any]) -> str:
    base, ship = rep["detectors"]["baseline"], rep["detectors"]["shipped"]
    lines = [
        "# Prompt-injection detector on real public data",
        "",
        f"Generated by `scripts/eval_real_injection.py` on {rep['date']} (seed {rep['seed']}). "
        "Detector = the deterministic sanitise + quarantine path (`guardrails/injection.py`); "
        "each dataset row is wrapped as a visible `<p>` and run through `quarantine()`. "
        "CIs are 95% Wilson intervals. `baseline` = the 15 rules that shipped before this "
        "evaluation; `shipped` = baseline + rules tuned on the dev splits only.",
        "",
        "## Held-out test splits",
        "",
        "| dataset | detector | recall (attacks flagged) | FPR (benign flagged) | precision | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for ds in ("deepset", "gandalf", "bipia", "pooled_injection", "jailbreak_cls"):
        for name, det in (("baseline", base), ("shipped", ship)):
            g = det["groups"].get(f"{ds}|test|all")
            if g is None:
                continue
            f1 = "n/a" if g["f1"] is None else f"{g['f1']:.3f}"
            lines.append(
                f"| {ds} | {name} | {_fmt(g['recall'])} | {_fmt(g['fpr'])} | "
                f"{_fmt(g['precision'])} | {f1} |"
            )
    lines += [
        "",
        "`pooled_injection` = deepset + gandalf + bipia. Gandalf and BIPIA contain attacks only, "
        "so their precision is trivially 100% and FPR is undefined; FPR comes from deepset's "
        "benign rows, jailbreak_cls's benign rows and the real web pages below.",
        "",
        "## deepset by language (heuristic stopword tag), test split",
        "",
        "| lang | detector | recall | FPR |",
        "|---|---|---|---|",
    ]
    for lang in ("en", "de", "unknown"):
        for name, det in (("baseline", base), ("shipped", ship)):
            g = det["groups"].get(f"deepset|test|{lang}")
            if g:
                lines.append(f"| {lang} | {name} | {_fmt(g['recall'])} | {_fmt(g['fpr'])} |")
    lines += [
        "",
        "## Dev splits (used for tuning; optimistic for `shipped`)",
        "",
        "| dataset | baseline recall | shipped recall | baseline FPR | shipped FPR |",
        "|---|---|---|---|---|",
    ]
    for ds in ("deepset", "gandalf", "bipia", "pooled_injection", "jailbreak_cls"):
        b, s = base["groups"].get(f"{ds}|dev|all"), ship["groups"].get(f"{ds}|dev|all")
        if b and s:
            lines.append(
                f"| {ds} | {_fmt(b['recall'])} | {_fmt(s['recall'])} | {_fmt(b['fpr'])} | "
                f"{_fmt(s['fpr'])} |"
            )
    wb, ws = base["web"], ship["web"]
    lines += [
        "",
        "## Real B2B web pages (benign), through SafeFetcher + quarantine",
        "",
        f"{ws['urls']} URLs in `fixtures/real/b2b_urls.txt`; {ws['fetched']} fetched, "
        f"{ws['failed']} failed (HTTP 403/429 bot blocks etc., listed in the JSON). "
        "Not used for tuning. A page counts if any finding fires anywhere, including in "
        "`<script>`/attribute text that the sanitiser drops anyway; visible-sentence redactions "
        "are what actually removes content from the Researcher's view.",
        "",
        "| detector | pages with any finding | visible sentences redacted |",
        "|---|---|---|",
        f"| baseline | {_fmt(wb['page_fpr'])} | {_fmt(wb['visible_sentence_fpr'])} |",
        f"| shipped | {_fmt(ws['page_fpr'])} | {_fmt(ws['visible_sentence_fpr'])} |",
    ]
    e = rep["end_to_end"]
    lines += [
        "",
        f"## End to end: {e['n']} real test-split attacks embedded in a fixture page",
        "",
        f"Sampled (seed {rep['seed']}) from deepset/gandalf/bipia test attacks: "
        + ", ".join(f"{k} {v}" for k, v in sorted(e["datasets"].items()))
        + ". Each attack is a visible paragraph on an otherwise benign company page, with a "
        "canary token inside its last sentence.",
        "",
        "Model-independent: how much of the attack text reaches the Researcher's context "
        "after quarantine:",
        "",
        "| detector | fully redacted | partly redacted | passed through verbatim |",
        "|---|---|---|---|",
    ]
    for name in ("baseline", "shipped"):
        x = e["exposure"][name]
        lines.append(
            f"| {name} | {x.get('blocked', 0)} | {x.get('partial', 0)} | {x.get('passed', 0)} |"
        )
    lines += [
        "",
        "Full mock pipeline (gullible mock model, see caveat), shipped detector:",
        "",
        "| guardrails | canary reached copy / forbidden tool / host | detector flagged page |",
        "|---|---|---|",
    ]
    for cfg, r in e["pipeline"].items():
        lines.append(f"| {cfg} | {r['compromised']}/{r['n']} | {r['detected']}/{r['n']} |")
    lines += [
        "",
        "Caveat: the mock only 'obeys' sentences that contain cue words and copies only "
        "canaries, URLs, emails and money amounts. Most real attacks ask for things it cannot "
        "act on (a poem, the password, an opinion), so the pipeline row is a lower bound on harm "
        "with a real model. The context-exposure table is the number to trust.",
        "",
        "## Examples (held-out data, shipped detector, excerpts <= 25 words)",
        "",
        "Missed attacks:",
        "",
    ]
    lines += [f"- [{m['dataset']}] {m['excerpt']}" for m in rep["examples"]["misses"]]
    lines += ["", "False positives:", ""]
    lines += [
        f"- [{f['source']}; {f['rule']}] {f['excerpt']}" for f in rep["examples"]["false_positives"]
    ]
    lines += [
        "",
        "## Datasets and licences",
        "",
        "| name | source | licence | kind |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {d['name']} | [{d['source']}]({d['url']}) | {d['licence']} | {d['kind']} |"
        for d in rep["datasets"]
    ]
    return "\n".join(lines) + "\n"
