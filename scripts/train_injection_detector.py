"""Train the model-based injection detector and score it once on held-out data.

    python scripts/train_injection_detector.py --skip-test   # fit + select on validation only
    python scripts/train_injection_detector.py               # ... then score TEST once

Needs the `detector` extra (scikit-learn) for fitting only; the runtime does not import it.
Uses the cached datasets/pages in data/real/ (run `make real-redteam` first, or pass --refresh).
Writes src/ai_sdr_crew/guardrails/injection_clf.json.gz and
reports/injection_classifier_eval.{json,md}.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import itertools
import json
import sys
import time
from typing import Any

import numpy as np
import sklearn
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ai_sdr_crew.config import PROJECT_ROOT
from ai_sdr_crew.evals import real_classifier as rc
from ai_sdr_crew.evals import real_injection as ri
from ai_sdr_crew.guardrails.injection_model import _TOKEN_RE as TOKEN_RE
from ai_sdr_crew.guardrails.injection_model import MODEL_PATH, InjectionClassifier

GRID = {
    "unit": ["sample", "block"],
    "C": [0.5, 2.0, 8.0],
    "class_weight": [None, "balanced"],
    "min_words": [1, 3],
}
WORD = {"analyzer": "word", "ngram_range": (1, 2), "max_features": 20000}
CHAR = {"analyzer": "char_wb", "ngram_range": (2, 5), "max_features": 50000}


def log(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def fit(texts: list[str], labels: list[int], cfg: dict[str, Any]) -> tuple[Any, Any, Any]:
    vw = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2, **WORD)
    vc = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2, **CHAR)
    x = hstack([vw.fit_transform(texts), vc.fit_transform(texts)]).tocsr()
    lr = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], max_iter=5000)
    lr.fit(x, np.asarray(labels))
    return vw, vc, lr


def sk_scorer(vw: Any, vc: Any, lr: Any, min_words: int = 1) -> rc.Scorer:
    def score(texts: Any) -> list[float]:
        if not texts:
            return []
        x = hstack([vw.transform(texts), vc.transform(texts)]).tocsr()
        p = lr.predict_proba(x)[:, 1]
        return [
            float(pi) if len(TOKEN_RE.findall(t)) >= min_words else 0.0
            for pi, t in zip(p, texts, strict=True)
        ]

    return score


def export(
    vw: Any, vc: Any, lr: Any, threshold: float, min_words: int, meta: dict[str, Any]
) -> dict[str, Any]:
    coef = lr.coef_[0]
    blocks = []
    offset = 0
    for vec, spec in ((vw, WORD), (vc, CHAR)):
        vocab: dict[str, int] = vec.vocabulary_
        terms = {
            t: [float(f"{vec.idf_[i]:.6g}"), float(f"{coef[offset + i]:.6g}")]
            for t, i in sorted(vocab.items())
        }
        offset += len(vocab)
        blocks.append(
            {
                "analyzer": spec["analyzer"],
                "ngram_range": list(spec["ngram_range"]),
                "sublinear_tf": True,
                "terms": terms,
            }
        )
    return {
        "blocks": blocks,
        "intercept": float(lr.intercept_[0]),
        "threshold": threshold,
        "min_words": min_words,
        "meta": meta,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-test", action="store_true", help="do not touch the TEST split")
    ap.add_argument("--refresh", action="store_true", help="re-download datasets and pages")
    ap.add_argument("--target-fpr", type=float, default=0.005, help="validation web-sentence FPR")
    ap.add_argument("--e2e-n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    samples = ri.load_samples(ri.download_datasets(refresh=args.refresh))
    fetched = ri.fetch_pages(ri.read_urls(), refresh=args.refresh)
    pages = [(u, str(r["html"])) for u, r in fetched if r.get("ok")]
    split = {s: [x for x in samples if rc.sample_split(x) == s] for s in ("train", "val", "test")}
    psplit = {s: [p for p in pages if rc.page_split(p[0]) == s] for s in ("train", "val", "test")}

    web_blocks = (b for _, h in psplit["train"] for b in rc.blocks_of(h))
    web_train: list[str] = list(dict.fromkeys(web_blocks))
    counts = {
        k: f"{len(split[k])} dataset rows "
        f"({sum(s.label for s in split[k])} attacks) + {len(psplit[k])} web pages"
        for k in split
    }
    counts["train"] += f" ({len(web_train)} unique visible blocks as benign)"
    log(counts)

    log("precomputing validation views ...")
    val_s = [rc.sample_view(s) for s in split["val"]]
    val_p = [rc.page_view(u, h) for u, h in psplit["val"]]

    results = []
    fitted: dict[tuple[Any, ...], tuple[Any, Any, Any]] = {}
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, combo, strict=True))
        texts: list[str] = []
        labels: list[int] = []
        for s in split["train"]:
            units = (
                [rc.sample_unit(s.text)]
                if cfg["unit"] == "sample"
                else rc.blocks_of(ri.as_page(s.text))
            )
            texts += units
            labels += [s.label] * len(units)
        texts += web_train
        labels += [0] * len(web_train)
        fit_key = (cfg["unit"], cfg["C"], cfg["class_weight"])
        if fit_key not in fitted:
            fitted[fit_key] = fit(texts, labels, cfg)
        model = fitted[fit_key]
        rc.attach_scores(val_s, val_p, sk_scorer(*model, min_words=cfg["min_words"]))
        t = rc.choose_threshold(val_p, args.target_fpr)
        m = rc.sample_metrics(val_s, t)["regex+classifier"]
        rec = m["pooled_injection"]["recall"]["value"]
        fpr = m["deepset"]["fpr"]["value"]
        log(f"{cfg} t={t:.3f} val pooled recall={rec:.4f} deepset FPR={fpr:.4f}")
        results.append((rec, -fpr, cfg, t, model))

    rec, _, cfg, t, model = max(results, key=lambda r: (r[0], r[1]))
    log(f"selected {cfg} threshold {t:.3f} (val pooled recall {rec:.4f})")
    meta = {
        "trained": dt.date.today().isoformat(),
        "config": cfg,
        "word": {**WORD, "ngram_range": list(WORD["ngram_range"])},
        "char": {**CHAR, "ngram_range": list(CHAR["ngram_range"])},
        "target_val_web_sentence_fpr": args.target_fpr,
        "sklearn": sklearn.__version__,
        "counts": counts,
    }
    art = export(*model, threshold=t, min_words=cfg["min_words"], meta=meta)
    MODEL_PATH.write_bytes(gzip.compress(json.dumps(art, ensure_ascii=False).encode(), 9, mtime=0))
    t0 = time.perf_counter()
    clf = InjectionClassifier.load(MODEL_PATH)
    load_ms = round(1000 * (time.perf_counter() - t0), 1)

    # The runtime re-implementation must reproduce scikit-learn's probabilities.
    probe = [b for v in val_s for b in v.blocks] + [b.text for p in val_p for b in p.blocks]
    sk = sk_scorer(*model, min_words=cfg["min_words"])(probe)
    diff = max(abs(a - clf.score(x)) for a, x in zip(sk, probe, strict=True))
    log(f"runtime vs scikit-learn max |dp| on {len(probe)} validation blocks: {diff:.2e}")
    assert diff < 1e-3, diff

    rc.attach_scores(val_s, val_p, lambda xs: [clf.score(x) for x in xs])
    val = {"samples": rc.sample_metrics(val_s, t), "web": rc.web_metrics(val_p, t)}
    size_kb = round(MODEL_PATH.stat().st_size / 1024)
    log(f"artefact {MODEL_PATH} {size_kb} KB")
    if args.skip_test:
        log(json.dumps({k: v["pooled_injection"]["recall"] for k, v in val["samples"].items()}))
        log(json.dumps({k: v["sentence_fpr"] for k, v in val["web"].items()}))
        return 0

    log("scoring TEST once ...")
    test = rc.evaluate_test(samples, psplit["test"], clf, e2e_n=args.e2e_n, seed=args.seed)
    rep = {
        "date": dt.date.today().isoformat(),
        "counts": counts,
        "grid_size": len(results),
        "grid": [
            {"config": c, "threshold": th, "val_pooled_recall": r, "val_deepset_fpr": -f}
            for r, f, c, th, _ in results
        ],
        "target_fpr": args.target_fpr,
        "model": {
            "description": "TF-IDF word 1-2-grams (20k) + char_wb 2-5-grams (50k), sublinear tf, "
            f"logistic regression (C={cfg['C']}, class_weight={cfg['class_weight']}, "
            f"training unit={cfg['unit']}, min {cfg['min_words']} words per scored block)",
            "threshold": t,
            "path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "size_kb": size_kb,
            "n_terms": sum(len(b["terms"]) for b in art["blocks"]),
            "load_ms": load_ms,
            "runtime_vs_sklearn_max_abs_diff": diff,
        },
        "test_pages": [u for u, _ in psplit["test"]],
        "validation": val,
        "test": test,
    }
    out = PROJECT_ROOT / "reports"
    (out / "injection_classifier_eval.json").write_text(json.dumps(rep, indent=2) + "\n", "utf-8")
    md = rc.report_markdown(rep)
    (out / "injection_classifier_eval.md").write_text(md, "utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
