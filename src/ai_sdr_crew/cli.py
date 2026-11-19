"""Command line interface: `sdr-crew <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ai_sdr_crew.config import PROJECT_ROOT, Settings

REPORTS = PROJECT_ROOT / "reports"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def cmd_run(args: argparse.Namespace) -> int:
    from ai_sdr_crew.evals.golden import build_env, load_golden
    from ai_sdr_crew.factory import make_pipeline
    from ai_sdr_crew.guardrails.citations import render
    from ai_sdr_crew.hitl.store import ApprovalStore
    from ai_sdr_crew.tracing.tracer import Tracer

    cases = load_golden()
    match = [c for c in cases if c.account.domain == args.domain]
    if not match:
        print(f"unknown domain {args.domain}; golden domains: {[c.account.domain for c in cases]}")
        return 2
    settings = Settings.from_env(prompt_version=args.prompt_version, live=args.live)
    pipeline = make_pipeline(settings, build_env(cases), tracer=Tracer(args.trace))
    r = pipeline.run(match[0].account)
    print(f"status: {r.status.value} {('(' + r.reason + ')') if r.reason else ''}")
    print(f"stages: {' -> '.join(r.stages)}")
    print(f"revisions: {r.revisions}  tokens: {r.usage.total_tokens}  cost est: ${r.cost_usd:.4f}")
    if r.sequence:
        for s in r.sequence.steps:
            print(f"\n--- email {s.step} (day +{s.delay_days}) | {s.subject}\n{render(s.body)}")
        print(f"\n--- linkedin\n{render(r.sequence.linkedin_note)}")
    if args.enqueue:
        item_id = ApprovalStore(args.db).enqueue(r)
        print(f"\nqueued as item {item_id}" if item_id else "\nnot queued (halted)")
    print(f"\ntrace: {args.trace}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from ai_sdr_crew.evals.runner import report_markdown, run_eval

    report = run_eval(args.prompt_version, live=args.live, trace_path=args.trace)
    stem = f"eval_{args.prompt_version}{'_live' if args.live else ''}"
    _write(args.out or REPORTS / f"{stem}.json", report.model_dump_json(indent=2))
    _write(REPORTS / f"{stem}.md", report_markdown(report))
    print(json.dumps(report.aggregates, indent=2))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from ai_sdr_crew.evals.regression import compare, gate_failures, regression_markdown
    from ai_sdr_crew.evals.runner import EvalReport

    base = EvalReport.model_validate_json(Path(args.baseline).read_text(encoding="utf-8"))
    cand = EvalReport.model_validate_json(Path(args.candidate).read_text(encoding="utf-8"))
    md = regression_markdown(base, cand)
    out = args.out or REPORTS / f"regression_{base.prompt_version}_{cand.prompt_version}.md"
    _write(Path(out), md)
    print(md)
    return 1 if args.fail_on_regression and gate_failures(compare(base, cand)) else 0


def cmd_redteam(args: argparse.Namespace) -> int:
    from ai_sdr_crew.evals.redteam import report_markdown, run_redteam

    outcomes = run_redteam()
    md = report_markdown(outcomes)
    _write(args.out or REPORTS / "redteam.md", md)
    print(md)
    breached = [o for o in outcomes if o.config == "all_on" and o.compromised]
    return 1 if breached else 0


def cmd_trace(args: argparse.Namespace) -> int:
    from ai_sdr_crew.tracing.viewer import load, render_tree, summarize

    records = load(args.file)
    print(summarize(records) if args.action == "summary" else render_tree(records, args.trace_id))
    return 0


def cmd_ab(args: argparse.Namespace) -> int:
    from ai_sdr_crew.experiments.bayes_ab import Arm, decide

    if args.action == "simulate":
        from ai_sdr_crew.experiments.simulate import aa_test, ab_test, calibration, report_markdown

        md = report_markdown(
            calibration(n_sims=args.sims * 10),
            aa_test(n_sims=args.sims),
            [ab_test(n_sims=args.sims, epsilon=e) for e in (0.002, 0.001, 0.0005)],
        )
        _write(args.out or REPORTS / "ab_simulation.md", md)
        print(md)
        return 0
    d = decide(
        Arm("control", args.control_sends, args.control_replies),
        Arm("variant", args.variant_sends, args.variant_replies),
        epsilon=args.epsilon,
    )
    print(f"P(variant beats control) = {d.prob_variant_better:.4f}")
    print(f"expected loss if we ship variant = {d.loss_ship_variant:.5f}")
    print(f"expected loss if we keep control = {d.loss_ship_control:.5f}")
    print(f"decision: {'STOP, ship ' + str(d.winner) if d.stop else 'keep testing'} ({d.reason})")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from ai_sdr_crew.evals.golden import build_env, load_golden
    from ai_sdr_crew.factory import make_pipeline
    from ai_sdr_crew.hitl.store import ApprovalStore
    from ai_sdr_crew.tracing.tracer import Tracer

    cases = load_golden()
    settings = Settings.from_env(prompt_version=args.prompt_version)
    pipeline = make_pipeline(settings, build_env(cases), tracer=Tracer(args.trace))
    store = ApprovalStore(args.db)
    queued = 0
    for c in cases:
        r = pipeline.run(c.account)
        queued += store.enqueue(r) is not None
        print(f"{c.account.domain:26} {r.status.value}")
    print(f"\nqueued {queued} sequences in {args.db}; trace in {args.trace}")
    print("next: sdr-crew serve  (then open http://127.0.0.1:8000)")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from ai_sdr_crew.hitl.app import create_app
    from ai_sdr_crew.hitl.store import ApprovalStore

    uvicorn.run(create_app(ApprovalStore(args.db)), host=args.host, port=args.port)
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from ai_sdr_crew.hitl.export import to_csv
    from ai_sdr_crew.hitl.store import ApprovalStore

    store = ApprovalStore(args.db)
    text = to_csv(store.approved(), args.format)
    _write(Path(args.out), text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sdr-crew", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    db_default = "data/queue.db"

    r = sub.add_parser("run", help="run the crew on one golden account")
    r.add_argument("domain")
    r.add_argument("--prompt-version", default="v2")
    r.add_argument("--live", action="store_true", help="call the Anthropic API (needs key)")
    r.add_argument("--trace", type=Path, default=Path("traces/run.jsonl"))
    r.add_argument("--enqueue", action="store_true")
    r.add_argument("--db", default=db_default)
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("eval", help="run the golden-set eval")
    e.add_argument("--prompt-version", default="v2")
    e.add_argument("--live", action="store_true")
    e.add_argument("--trace", type=Path, default=None)
    e.add_argument("--out", type=Path, default=None)
    e.set_defaults(func=cmd_eval)

    c = sub.add_parser("compare", help="regression report between two eval JSON files")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--out", default=None)
    c.add_argument("--fail-on-regression", action="store_true")
    c.set_defaults(func=cmd_compare)

    rt = sub.add_parser("redteam", help="run the prompt-injection red-team matrix")
    rt.add_argument("--out", type=Path, default=None)
    rt.set_defaults(func=cmd_redteam)

    t = sub.add_parser("trace", help="view a JSONL trace")
    t.add_argument("action", choices=["view", "summary"])
    t.add_argument("file", type=Path)
    t.add_argument("--trace-id", default=None)
    t.set_defaults(func=cmd_trace)

    ab = sub.add_parser("ab", help="Bayesian A/B analysis or simulation")
    ab.add_argument("action", choices=["analyze", "simulate"])
    ab.add_argument("--control-sends", type=int, default=0)
    ab.add_argument("--control-replies", type=int, default=0)
    ab.add_argument("--variant-sends", type=int, default=0)
    ab.add_argument("--variant-replies", type=int, default=0)
    ab.add_argument("--epsilon", type=float, default=0.001)
    ab.add_argument("--sims", type=int, default=1000)
    ab.add_argument("--out", type=Path, default=None)
    ab.set_defaults(func=cmd_ab)

    d = sub.add_parser("demo", help="run all golden accounts and fill the approval queue")
    d.add_argument("--prompt-version", default="v2")
    d.add_argument("--db", default=db_default)
    d.add_argument("--trace", type=Path, default=Path("traces/demo.jsonl"))
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="start the approval UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--db", default=db_default)
    s.set_defaults(func=cmd_serve)

    x = sub.add_parser("export", help="export approved sequences to CSV")
    x.add_argument("--format", choices=["instantly", "smartlead"], default="instantly")
    x.add_argument("--out", default="exports/approved.csv")
    x.add_argument("--db", default=db_default)
    x.set_defaults(func=cmd_export)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rc: int = args.func(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())
