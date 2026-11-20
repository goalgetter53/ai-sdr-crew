"""Second, model-based prompt-injection detector: TF-IDF word + char n-grams, logistic regression.

Trained offline with scikit-learn (`make train-detector`, see scripts/train_injection_detector.py)
on the TRAIN splits of public injection datasets plus sentences from real B2B pages; the threshold
was picked on the VALIDATION split. The runtime below re-implements scikit-learn's analyzers and
TF-IDF weighting in plain Python, so serving needs no scikit-learn, no pickle and no network: the
artefact is a gzipped JSON of {term: [idf, coef]} plus intercept and threshold.

Loading is lazy and optional: if the artefact is missing or unreadable, or SDR_INJECTION_CLASSIFIER
is "0"/"off"/"false", `default_classifier()` returns None and quarantine falls back to regex only.
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).with_name("injection_clf.json.gz")
ENV_SWITCH = "SDR_INJECTION_CLASSIFIER"
_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")  # scikit-learn's default token_pattern
_WS2_RE = re.compile(r"\s\s+")


def word_ngrams(text: str, lo: int, hi: int) -> list[str]:
    """Same terms as TfidfVectorizer(analyzer='word', lowercase=True, ngram_range=(lo, hi))."""
    toks = _TOKEN_RE.findall(text.lower())
    out: list[str] = []
    for n in range(lo, hi + 1):
        out.extend(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return out


def char_wb_ngrams(text: str, lo: int, hi: int) -> list[str]:
    """Same terms as TfidfVectorizer(analyzer='char_wb', lowercase=True, ngram_range=(lo, hi))."""
    out: list[str] = []
    for word in _WS2_RE.sub(" ", text.lower()).split():
        w = f" {word} "
        for n in range(lo, hi + 1):
            if len(w) <= n:
                out.append(w)  # scikit-learn counts a short word once, then stops
                break
            out.extend(w[i : i + n] for i in range(len(w) - n + 1))
    return out


@dataclass(frozen=True)
class _Block:
    analyzer: str  # "word" | "char_wb"
    lo: int
    hi: int
    sublinear_tf: bool
    terms: dict[str, tuple[float, float]]  # term -> (idf, coef)

    def terms_of(self, text: str) -> list[str]:
        fn = word_ngrams if self.analyzer == "word" else char_wb_ngrams
        return fn(text, self.lo, self.hi)

    def logit(self, text: str) -> float:
        counts = Counter(t for t in self.terms_of(text) if t in self.terms)
        if not counts:
            return 0.0
        num, sq = 0.0, 0.0
        for term, c in counts.items():
            idf, coef = self.terms[term]
            w = (1.0 + math.log(c) if self.sublinear_tf else float(c)) * idf
            num += w * coef
            sq += w * w
        return num / math.sqrt(sq)


@dataclass(frozen=True)
class InjectionClassifier:
    blocks: tuple[_Block, ...]
    intercept: float
    threshold: float
    meta: dict[str, Any]
    min_words: int = 1  # blocks with fewer word tokens are never flagged ("Python", "Previous")

    def score(self, text: str) -> float:
        """P(injection) for one block of visible text (0.0 if shorter than `min_words`)."""
        if len(_TOKEN_RE.findall(text)) < self.min_words:
            return 0.0
        z = self.intercept + sum(b.logit(text) for b in self.blocks)
        return 1.0 / (1.0 + math.exp(-z)) if z > -700 else 0.0

    def flags(self, text: str) -> bool:
        return self.score(text) >= self.threshold

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InjectionClassifier:
        blocks = tuple(
            _Block(
                analyzer=str(b["analyzer"]),
                lo=int(b["ngram_range"][0]),
                hi=int(b["ngram_range"][1]),
                sublinear_tf=bool(b["sublinear_tf"]),
                terms={str(k): (float(v[0]), float(v[1])) for k, v in b["terms"].items()},
            )
            for b in d["blocks"]
        )
        return cls(
            blocks,
            float(d["intercept"]),
            float(d["threshold"]),
            dict(d.get("meta", {})),
            int(d.get("min_words", 1)),
        )

    @classmethod
    def load(cls, path: Path) -> InjectionClassifier:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def enabled_by_env() -> bool:
    return os.environ.get(ENV_SWITCH, "1").strip().lower() not in {"0", "off", "false", "no"}


@lru_cache(maxsize=4)
def _load_cached(path: Path) -> InjectionClassifier | None:
    if not path.exists():
        log.warning("injection classifier artefact missing at %s; regex layer only", path)
        return None
    try:
        return InjectionClassifier.load(path)
    except (OSError, ValueError, KeyError, TypeError) as e:
        log.warning("injection classifier failed to load (%s); regex layer only", e)
        return None


def default_classifier(path: Path | None = None) -> InjectionClassifier | None:
    """The shipped classifier, loaded once on first use; None if disabled or unavailable."""
    if not enabled_by_env():
        return None
    return _load_cached(path or MODEL_PATH)
