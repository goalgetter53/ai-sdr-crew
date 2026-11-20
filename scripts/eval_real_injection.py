"""Evaluate the deterministic injection defence on real public attack data and real web pages.

    python scripts/eval_real_injection.py            # uses cache in data/real/ if present
    python scripts/eval_real_injection.py --refresh  # re-download datasets and pages

Needs network on first run and the `real` extra (pyarrow): pip install -e ".[dev,real]".
Writes reports/real_injection_eval.json and reports/real_injection_eval.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys

from ai_sdr_crew.config import PROJECT_ROOT
from ai_sdr_crew.evals import real_injection as ri


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-download datasets and pages")
    ap.add_argument("--e2e-n", type=int, default=100, help="attacks for the end-to-end run")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    samples = ri.load_samples(ri.download_datasets(refresh=args.refresh))
    print(f"{len(samples)} dataset rows", file=sys.stderr)
    pages = ri.fetch_pages(ri.read_urls(), refresh=args.refresh)
    print(f"{sum(1 for _, r in pages if r.get('ok'))}/{len(pages)} pages fetched", file=sys.stderr)
    rep = ri.build_report(
        samples, pages, e2e_n=args.e2e_n, seed=args.seed, run_date=dt.date.today().isoformat()
    )
    out = PROJECT_ROOT / "reports"
    (out / "real_injection_eval.json").write_text(json.dumps(rep, indent=2) + "\n", "utf-8")
    md = ri.report_markdown(rep)
    (out / "real_injection_eval.md").write_text(md, "utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
