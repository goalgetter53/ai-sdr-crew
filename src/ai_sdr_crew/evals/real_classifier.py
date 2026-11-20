"""Train/validation/test protocol and held-out scoring for the model-based injection detector.

Splits (fixed, hash-based, independent of any model):
- dataset rows: the datasets' own test splits stay TEST. Their train(+validation) rows ("dev" in
  real_injection.py) are split 80/20 into TRAIN / VALIDATION by a hash of (dataset, text).
- real B2B pages: split BY PAGE (hash of the URL) into TRAIN 60% / VALIDATION 20% / TEST 20%, so no
  sentence of a test page is ever seen in training or threshold selection.

The classifier is fitted on TRAIN (scripts/train_injection_detector.py), its hyperparameters and
threshold are chosen on VALIDATION, and TEST is scored once by `evaluate_test` below through the
same `quarantine()` call the Researcher's fetch tool uses.
"""

from __future__ import annotations

import hashlib
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from html import escape
from typing import Any

from ai_sdr_crew.config import Guardrails
from ai_sdr_crew.evals.real_injection import _PAGE as E2E_PAGE
from ai_sdr_crew.evals.real_injection import (
    INJECTION_DATASETS,
    Confusion,
    Sample,
    _fmt,
    _pick,
    _rate,
    as_page,
    excerpt,
    sample_attacks,
    with_canary,
)
from ai_sdr_crew.evals.redteam import CONFIGS, RedTeamPage, run_html
from ai_sdr_crew.guardrails.injection import (
    RULES,
    detect,
    extract,
    normalize,
    quarantine,
    split_sentences,
)
from ai_sdr_crew.guardrails.injection_model import InjectionClassifier

Scorer = Callable[[Sequence[str]], list[float]]


def _bucket(key: str, mod: int) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % mod


def sample_split(s: Sample) -> str:
    """'train' | 'val' | 'test'."""
    if s.split == "test":
        return "test"
    return "val" if _bucket(f"{s.dataset}\x00{s.text}", 5) == 0 else "train"


def page_split(url: str) -> str:
    b = _bucket(url, 10)
    return "train" if b < 6 else ("val" if b < 8 else "test")


def blocks_of(html: str) -> list[str]:
    """Visible text blocks exactly as quarantine() iterates them (one per non-empty line)."""
    visible, _ = extract(html)
    return [b.strip() for b in normalize(visible).split("\n") if b.strip()]


def sample_unit(text: str) -> str:
    """Whole dataset row as one training unit (whitespace collapsed)."""
    return " ".join(normalize(text).split())


# ---------------------------------------------------------------------------------------------
# precomputed views so thresholds can be swept without re-running quarantine


@dataclass
class SampleView:
    sample: Sample
    blocks: list[str]
    regex: bool
    scores: list[float] = field(default_factory=list)

    def flagged(self, t: float, use_regex: bool = True, use_model: bool = True) -> bool:
        return (use_regex and self.regex) or (use_model and any(x >= t for x in self.scores))


@dataclass
class BlockView:
    text: str
    regex_hits: list[bool]  # per sentence
    score: float = 0.0


@dataclass
class PageView:
    url: str
    blocks: list[BlockView]

    @property
    def n_sentences(self) -> int:
        return sum(len(b.regex_hits) for b in self.blocks)

    def redacted(self, t: float, use_regex: bool = True, use_model: bool = True) -> int:
        n = 0
        for b in self.blocks:
            if use_model and b.score >= t:
                n += len(b.regex_hits)
            elif use_regex:
                n += sum(b.regex_hits)
        return n


def sample_view(s: Sample) -> SampleView:
    regex = quarantine(as_page(s.text), "dataset", "s", RULES, use_classifier=False).flagged
    return SampleView(s, blocks_of(as_page(s.text)), regex)


def page_view(url: str, html: str) -> PageView:
    blocks = [
        BlockView(b, [bool(detect(x, "visible", RULES)) for x in split_sentences(b)])
        for b in blocks_of(html)
    ]
    return PageView(url, [b for b in blocks if b.regex_hits])


def attach_scores(samples: list[SampleView], pages: list[PageView], scorer: Scorer) -> None:
    texts = [b for s in samples for b in s.blocks]
    texts += [b.text for p in pages for b in p.blocks]
    it = iter(scorer(texts))
    for s in samples:
        s.scores = [next(it) for _ in s.blocks]
    for p in pages:
        for b in p.blocks:
            b.score = next(it)


# ---------------------------------------------------------------------------------------------
# metrics


DETECTORS: dict[str, tuple[bool, bool]] = {  # name -> (use_regex, use_model)
    "regex": (True, False),
    "classifier": (False, True),
    "regex+classifier": (True, True),
}
GROUPS = ("deepset", "gandalf", "bipia", "pooled_injection", "jailbreak_cls")


def sample_metrics(views: list[SampleView], t: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for det, (ur, um) in DETECTORS.items():
        groups: dict[str, Confusion] = {}
        for v in views:
            f = v.flagged(t, ur, um)
            keys = [v.sample.dataset]
            if v.sample.dataset in INJECTION_DATASETS:
                keys.append("pooled_injection")
            for k in keys:
                groups.setdefault(k, Confusion()).add(v.sample.label, f)
        out[det] = {k: groups[k].metrics() for k in GROUPS if k in groups}
    return out


def web_metrics(pages: list[PageView], t: float) -> dict[str, dict[str, Any]]:
    n_sent = sum(p.n_sentences for p in pages)
    out: dict[str, dict[str, Any]] = {}
    for det, (ur, um) in DETECTORS.items():
        red = [p.redacted(t, ur, um) for p in pages]
        out[det] = {
            "pages": len(pages),
            "sentence_fpr": _rate(sum(red), n_sent),
            "page_fpr": _rate(sum(r > 0 for r in red), len(pages)),
        }
    return out


def choose_threshold(
    pages: list[PageView], target_fpr: float, grid: Sequence[float] | None = None
) -> float:
    """Lowest threshold whose COMBINED (regex OR model) web-sentence FPR on `pages` <= target."""
    n = sum(p.n_sentences for p in pages)
    for t in grid or [i / 1000 for i in range(50, 1000)]:
        if n == 0 or sum(p.redacted(t) for p in pages) / n <= target_fpr:
            return t
    return 1.0


# ---------------------------------------------------------------------------------------------
# held-out test: through quarantine() itself


def evaluate_test(
    samples: list[Sample],
    pages: list[tuple[str, str]],
    clf: InjectionClassifier,
    e2e_n: int = 100,
    seed: int = 7,
) -> dict[str, Any]:
    test = [s for s in samples if sample_split(s) == "test"]
    views: list[SampleView] = []
    for s in test:
        v = sample_view(s)
        q = quarantine(as_page(s.text), "dataset", "s", RULES, classifier=clf)
        v.scores = [clf.score(b) for b in v.blocks]
        assert q.flagged == v.flagged(clf.threshold), "sweep view diverged from quarantine()"
        views.append(v)
    pviews: list[PageView] = []
    lat_regex: list[float] = []
    lat_both: list[float] = []
    redacted_q = 0
    for url, html in pages:
        pv = page_view(url, html)
        for b in pv.blocks:
            b.score = clf.score(b.text)
        pviews.append(pv)
        t0 = time.perf_counter()
        quarantine(html, url, "web-1", RULES, use_classifier=False)
        t1 = time.perf_counter()
        q = quarantine(html, url, "web-1", RULES, classifier=clf)
        t2 = time.perf_counter()
        lat_regex.append(1000 * (t1 - t0))
        lat_both.append(1000 * (t2 - t1))
        redacted_q += q.redacted_sentences
    assert redacted_q == sum(p.redacted(clf.threshold) for p in pviews)
    t = clf.threshold
    fp_web = [
        {"url": p.url, "excerpt": excerpt(b.text), "score": round(b.score, 3)}
        for p in pviews
        for b in p.blocks
        if b.score >= t
    ]
    misses = [
        {"dataset": v.sample.dataset, "excerpt": excerpt(v.sample.text)}
        for v in views
        if v.sample.label == 1 and not v.flagged(t)
    ]
    fp_prompts = [
        {"dataset": v.sample.dataset, "excerpt": excerpt(v.sample.text)}
        for v in views
        if v.sample.label == 0 and v.flagged(t) and not v.regex
    ]
    return {
        "threshold": t,
        "samples": sample_metrics(views, t),
        "web": web_metrics(pviews, t),
        "web_sentences": sum(p.n_sentences for p in pviews),
        "latency_ms_per_page": {
            "regex": _lat(lat_regex, pages),
            "regex+classifier": _lat(lat_both, pages),
        },
        "examples": {
            "missed_attacks": misses,
            "classifier_fp_web_blocks": fp_web,
            "classifier_fp_prompts": fp_prompts,
        },
        "end_to_end": end_to_end(samples, clf, e2e_n, seed),
    }


def _lat(ms: list[float], pages: list[tuple[str, str]]) -> dict[str, float]:
    if not ms:
        return {}
    srt = sorted(ms)
    return {
        "median": round(statistics.median(ms), 2),
        "p95": round(srt[min(len(srt) - 1, int(0.95 * len(srt)))], 2),
        "max": round(srt[-1], 2),
        "mean_kb": round(sum(len(h) for _, h in pages) / len(pages) / 1024, 1),
    }


def _exposure(attack: str, clf: InjectionClassifier | None) -> str:
    q = quarantine(
        as_page(attack), "e2e", "web-1", RULES, use_classifier=clf is not None, classifier=clf
    )
    sentences = split_sentences(normalize(attack))
    survived = sum(s.replace("<", "‹").replace(">", "›") in q.visible_text for s in sentences)
    if survived == 0:
        return "blocked"
    return "passed" if survived == len(sentences) else "partial"


def end_to_end(
    samples: list[Sample], clf: InjectionClassifier, n: int, seed: int
) -> dict[str, Any]:
    """Same 100 attacks as reports/real_injection_eval.md (same sampler, same seed)."""
    attacks = sample_attacks(samples, n, seed)
    setups: dict[str, Guardrails] = {
        "all_on, regex only": replace(CONFIGS["all_on"], injection_classifier=False),
        "all_on, regex + classifier": CONFIGS["all_on"],
        "all_off": CONFIGS["all_off"],
    }
    exposure: dict[str, dict[str, int]] = {"regex": {}, "regex+classifier": {}}
    for a in attacks:
        for name, c in (("regex", None), ("regex+classifier", clf)):
            e = _exposure(a.text, c)
            exposure[name][e] = exposure[name].get(e, 0) + 1
    pipeline: dict[str, dict[str, int]] = {}
    for label, g in setups.items():
        pwned = detected = 0
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
            html = E2E_PAGE.format(company=page.company, attack=escape(with_canary(a.text, canary)))
            o = run_html(page, html, "all_on", g)
            pwned += o.compromised
            detected += o.detected
        pipeline[label] = {"n": len(attacks), "compromised": pwned, "detected": detected}
    by_ds: dict[str, int] = {}
    for a in attacks:
        by_ds[a.dataset] = by_ds.get(a.dataset, 0) + 1
    return {"n": len(attacks), "datasets": by_ds, "exposure": exposure, "pipeline": pipeline}


# ---------------------------------------------------------------------------------------------
# report


def report_markdown(rep: dict[str, Any]) -> str:
    t = rep["test"]
    m = rep["model"]
    lines = [
        "# Model-based injection detector (TF-IDF + logistic regression) on real data",
        "",
        f"Generated by `scripts/train_injection_detector.py` on {rep['date']}. Every number below "
        "is script output. 95% Wilson CIs. Detectors: `regex` = the shipped rule set "
        "(guardrails/injection.py); `classifier` = the model alone; `regex+classifier` = a "
        "sentence is redacted if either fires (what now ships).",
        "",
        "## Protocol",
        "",
        f"- Train: {rep['counts']['train']}. Validation: {rep['counts']['val']}. "
        f"Test: {rep['counts']['test']}.",
        "- Dataset rows: each dataset's own test split is TEST; its train(+validation) rows are "
        "split 80/20 into TRAIN / VALIDATION by a hash of the text. Real B2B pages are split BY "
        "PAGE (hash of URL) 60/20/20, so test pages are never seen in training or threshold "
        "selection.",
        f"- Model: {m['description']}. Hyperparameters chosen on validation from "
        f"{rep['grid_size']} configurations (max validation pooled-injection recall of "
        "regex+classifier at the threshold rule below); fitted on TRAIN only.",
        f"- Threshold rule (stated before looking at test): the lowest threshold at which "
        f"regex+classifier redacts <= {100 * rep['target_fpr']:.1f}% of visible sentences on "
        f"the VALIDATION pages. Chosen threshold: {m['threshold']:.3f}.",
        "- TEST was scored once, through `quarantine()` itself (the same call the fetch tool "
        "makes). The regex rules were tuned earlier on all non-test rows (train+validation "
        "here), so regex numbers on validation are optimistic; test is clean for both.",
        f"- Artefact: `{m['path']}` ({m['size_kb']} KB gzipped JSON, {m['n_terms']} terms; no "
        "pickle, no scikit-learn at runtime).",
        "",
        "## Held-out test: attacks and benign prompts",
        "",
        "| dataset | detector | recall | FPR (benign prompts) | precision | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for ds in GROUPS:
        for det in DETECTORS:
            g = t["samples"][det].get(ds)
            if g is None:
                continue
            f1 = "n/a" if g["f1"] is None else f"{g['f1']:.3f}"
            lines.append(
                f"| {ds} | {det} | {_fmt(g['recall'])} | {_fmt(g['fpr'])} | "
                f"{_fmt(g['precision'])} | {f1} |"
            )
    w = t["web"]
    lines += [
        "",
        "`pooled_injection` = deepset + gandalf + bipia (its benign rows are deepset's). Gandalf "
        "and BIPIA are attacks only, so their precision is trivially 100%.",
        "",
        f"## Held-out test: real B2B pages ({w['regex']['pages']} pages, "
        f"{t['web_sentences']} visible sentences)",
        "",
        "| detector | visible sentences redacted (FPR) | pages with any visible redaction |",
        "|---|---|---|",
    ]
    for det in DETECTORS:
        lines.append(f"| {det} | {_fmt(w[det]['sentence_fpr'])} | {_fmt(w[det]['page_fpr'])} |")
    v = rep["validation"]
    lines += [
        "",
        "## Validation (used for model selection and the threshold; optimistic)",
        "",
        "| detector | pooled_injection recall | deepset benign FPR | web sentence FPR |",
        "|---|---|---|---|",
    ]
    for det in DETECTORS:
        s = v["samples"][det]
        lines.append(
            f"| {det} | {_fmt(s['pooled_injection']['recall'])} | "
            f"{_fmt(s['deepset']['fpr'])} | {_fmt(v['web'][det]['sentence_fpr'])} |"
        )
    lat = t["latency_ms_per_page"]
    e = t["end_to_end"]
    lines += [
        "",
        "## Latency (quarantine() per held-out page, this machine, 1 CPU thread)",
        "",
        "| path | median ms | p95 ms | max ms |",
        "|---|---|---|---|",
        f"| regex | {lat['regex']['median']} | {lat['regex']['p95']} | {lat['regex']['max']} |",
        f"| regex+classifier | {lat['regex+classifier']['median']} | "
        f"{lat['regex+classifier']['p95']} | {lat['regex+classifier']['max']} |",
        "",
        f"Mean page size {lat['regex']['mean_kb']} KB of HTML. Model load (once per process): "
        f"{m['load_ms']} ms.",
        "",
        f"## End to end: the same {e['n']} real test attacks as `real_injection_eval.md`",
        "",
        "Sampled (seed 7) from deepset/gandalf/bipia test attacks: "
        + ", ".join(f"{k} {n}" for k, n in sorted(e["datasets"].items()))
        + ". Each attack is a visible paragraph on a benign fixture company page with a canary "
        "in its last sentence.",
        "",
        "Attack text reaching the Researcher's context after quarantine:",
        "",
        "| detector | fully redacted | partly redacted | passed through verbatim |",
        "|---|---|---|---|",
    ]
    for det in ("regex", "regex+classifier"):
        x = e["exposure"][det]
        lines.append(
            f"| {det} | {x.get('blocked', 0)} | {x.get('partial', 0)} | {x.get('passed', 0)} |"
        )
    lines += [
        "",
        "Full mock pipeline (gullible mock model; it only acts on canaries, links, emails and "
        "amounts, so this is a lower bound on harm with a real model):",
        "",
        "| setup | canary in copy / forbidden tool / host | detector flagged page |",
        "|---|---|---|",
    ]
    for name, r in e["pipeline"].items():
        lines.append(f"| {name} | {r['compromised']}/{r['n']} | {r['detected']}/{r['n']} |")
    ex = t["examples"]
    lines += ["", "## Examples (held-out, regex+classifier, excerpts <= 25 words)", ""]
    lines += ["Missed attacks (8 of " + str(len(ex["missed_attacks"])) + "):", ""]
    lines += [f"- [{x['dataset']}] {x['excerpt']}" for x in _pick(ex["missed_attacks"], 8, 7)]
    lines += [
        "",
        "Classifier-only false positives on test web blocks "
        f"({len(ex['classifier_fp_web_blocks'])}):",
        "",
    ]
    lines += [
        f"- [{x['url']}; p={x['score']}] {x['excerpt']}" for x in ex["classifier_fp_web_blocks"][:8]
    ]
    lines += [
        "",
        "Classifier-only false positives on benign test prompts "
        f"({len(ex['classifier_fp_prompts'])}, first 5):",
        "",
    ]
    lines += [f"- [{x['dataset']}] {x['excerpt']}" for x in ex["classifier_fp_prompts"][:5]]
    return "\n".join(lines) + "\n"
