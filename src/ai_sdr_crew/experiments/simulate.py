"""Simulation-based validation of the Bayesian A/B module.

1. Calibration: when true rates are drawn from the prior, events predicted with probability x
   should happen ~x of the time. Large gaps would mean a bug in the posterior maths.
2. A/A false-positive rate: with equal true rates, how often does 'P(variant better) > 0.95'
   declare a winner at a fixed horizon (should be ~5%) vs. when peeking after every batch
   (inflated)? This is why the stopping rule is based on expected loss, not on the probability.
3. A/B decision quality: with a real lift, how often the expected-loss rule picks the right arm,
   how many sends it needs, and the realised regret when it is wrong.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ai_sdr_crew.experiments.bayes_ab import Arm, decide, prob_greater


@dataclass(frozen=True)
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_predicted: float
    observed: float


@dataclass(frozen=True)
class CalibrationResult:
    n_sims: int
    bins: list[CalibrationBin]
    max_abs_gap: float
    brier: float


def calibration(
    n_sims: int = 10000, sends: int = 1000, prior: tuple[int, int] = (2, 40), seed: int = 7
) -> CalibrationResult:
    rng = np.random.default_rng(seed)
    pa = rng.beta(prior[0], prior[1], n_sims)
    pb = rng.beta(prior[0], prior[1], n_sims)
    ra = rng.binomial(sends, pa)
    rb = rng.binomial(sends, pb)
    pred = np.array(
        [
            prob_greater(
                prior[0] + int(a),
                prior[1] + sends - int(a),
                prior[0] + int(b),
                prior[1] + sends - int(b),
            )
            for a, b in zip(ra, rb, strict=True)
        ]
    )
    truth = (pb > pa).astype(float)
    bins: list[CalibrationBin] = []
    edges = np.linspace(0, 1, 11)
    gaps: list[float] = []
    for lo, hi in itertools.pairwise(edges):
        mask = (pred >= lo) & ((pred < hi) if hi < 1 else (pred <= hi))
        n = int(mask.sum())
        if n == 0:
            continue
        b = CalibrationBin(
            float(lo), float(hi), n, float(pred[mask].mean()), float(truth[mask].mean())
        )
        bins.append(b)
        if n >= 50:
            gaps.append(abs(b.mean_predicted - b.observed))
    return CalibrationResult(
        n_sims=n_sims,
        bins=bins,
        max_abs_gap=max(gaps) if gaps else 0.0,
        brier=float(((pred - truth) ** 2).mean()),
    )


@dataclass(frozen=True)
class AAResult:
    n_sims: int
    rate: float
    max_sends: int
    peeks: int
    fixed_horizon_fpr: float
    peeking_fpr: float
    loss_rule_stop_rate: float
    loss_rule_mean_sends: float


def _paths(rng: np.random.Generator, rate: float, n_sims: int, max_sends: int) -> np.ndarray:
    return np.cumsum(rng.random((n_sims, max_sends)) < rate, axis=1)


def aa_test(
    n_sims: int = 1000,
    rate: float = 0.04,
    max_sends: int = 3000,
    peek_every: int = 150,
    threshold: float = 0.95,
    epsilon: float = 0.001,
    seed: int = 11,
) -> AAResult:
    rng = np.random.default_rng(seed)
    ca = _paths(rng, rate, n_sims, max_sends)
    cb = _paths(rng, rate, n_sims, max_sends)
    checkpoints = list(range(peek_every, max_sends + 1, peek_every))
    fixed_fp = peek_fp = stops = 0
    stop_sends: list[int] = []
    for s in range(n_sims):
        peeked = stopped = False
        for n in checkpoints:
            a, b = int(ca[s, n - 1]), int(cb[s, n - 1])
            p = prob_greater(1 + a, 1 + n - a, 1 + b, 1 + n - b)
            if p > threshold:
                peeked = True
            if not stopped:
                d = decide(Arm("control", n, a), Arm("variant", n, b), epsilon=epsilon)
                if d.stop:
                    stopped = True
                    stop_sends.append(n)
            if n == max_sends and p > threshold:
                fixed_fp += 1
        peek_fp += peeked
        stops += stopped
    return AAResult(
        n_sims=n_sims,
        rate=rate,
        max_sends=max_sends,
        peeks=len(checkpoints),
        fixed_horizon_fpr=fixed_fp / n_sims,
        peeking_fpr=peek_fp / n_sims,
        loss_rule_stop_rate=stops / n_sims,
        loss_rule_mean_sends=float(np.mean(stop_sends)) if stop_sends else float("nan"),
    )


@dataclass(frozen=True)
class ABResult:
    n_sims: int
    control_rate: float
    variant_rate: float
    epsilon: float
    stop_rate: float
    correct_rate_when_stopped: float
    mean_sends_at_stop: float
    mean_regret_points: float


def ab_test(
    n_sims: int = 1000,
    control_rate: float = 0.04,
    variant_rate: float = 0.05,
    max_sends: int = 3000,
    peek_every: int = 150,
    epsilon: float = 0.001,
    seed: int = 13,
) -> ABResult:
    rng = np.random.default_rng(seed)
    ca = _paths(rng, control_rate, n_sims, max_sends)
    cb = _paths(rng, variant_rate, n_sims, max_sends)
    better = "variant" if variant_rate > control_rate else "control"
    gap = abs(variant_rate - control_rate)
    stops = correct = 0
    sends: list[int] = []
    regret: list[float] = []
    for s in range(n_sims):
        winner = None
        for n in range(peek_every, max_sends + 1, peek_every):
            a, b = int(ca[s, n - 1]), int(cb[s, n - 1])
            d = decide(Arm("control", n, a), Arm("variant", n, b), epsilon=epsilon)
            if d.stop:
                winner = d.winner
                sends.append(n)
                break
        if winner is None:
            continue
        stops += 1
        correct += winner == better
        regret.append(0.0 if winner == better else gap)
    return ABResult(
        n_sims=n_sims,
        control_rate=control_rate,
        variant_rate=variant_rate,
        epsilon=epsilon,
        stop_rate=stops / n_sims,
        correct_rate_when_stopped=correct / stops if stops else float("nan"),
        mean_sends_at_stop=float(np.mean(sends)) if sends else float("nan"),
        mean_regret_points=float(np.mean(regret)) if regret else float("nan"),
    )


def report_markdown(cal: CalibrationResult, aa: AAResult, abs_: list[ABResult]) -> str:
    lines = [
        "# Bayesian A/B module: simulation report",
        "",
        "All numbers below are produced by `sdr-crew ab simulate` (seeded, reproducible). They "
        "validate the statistics code on synthetic data; they are not campaign results.",
        "",
        f"## 1. Calibration of P(variant beats control) ({cal.n_sims} simulated tests)",
        "",
        "| predicted bin | n | mean predicted | observed frequency |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {b.lo:.1f}-{b.hi:.1f} | {b.n} | {b.mean_predicted:.3f} | {b.observed:.3f} |"
        for b in cal.bins
    ]
    lines += [
        "",
        f"Max |predicted - observed| over bins with n >= 50: **{cal.max_abs_gap:.3f}**; "
        f"Brier score {cal.brier:.4f}.",
        "",
        f"## 2. A/A false-positive rate ({aa.n_sims} simulations, true rate {aa.rate:.1%} both "
        f"arms, up to {aa.max_sends} sends/arm, {aa.peeks} looks)",
        "",
        "| decision rule | false-positive rate |",
        "|---|---|",
        f"| P(better) > 0.95, single look at the end | **{aa.fixed_horizon_fpr:.1%}** |",
        f"| P(better) > 0.95, checked at every look (peeking) | **{aa.peeking_fpr:.1%}** |",
        "",
        f"The expected-loss rule stopped in {aa.loss_rule_stop_rate:.1%} of A/A runs (mean "
        f"{aa.loss_rule_mean_sends:.0f} sends/arm). In an A/A test either choice costs nothing, "
        "which is exactly what the loss rule optimises: it bounds the *cost* of a wrong call "
        "to epsilon instead of pretending to control a false-positive rate under peeking.",
        "",
        f"## 3. Decision quality with a real lift ({abs_[0].control_rate:.1%} -> "
        f"{abs_[0].variant_rate:.1%}, {abs_[0].n_sims} simulations per row, cap "
        "3000 sends/arm)",
        "",
        "| epsilon (reply-rate points) | stopped | picked better arm | mean sends/arm at stop "
        "| mean regret (points) |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {ab.epsilon * 100:.2f} | {ab.stop_rate:.1%} | {ab.correct_rate_when_stopped:.1%} | "
        f"{ab.mean_sends_at_stop:.0f} | {ab.mean_regret_points * 100:.3f} |"
        for ab in abs_
    ]
    lines += [
        "",
        "Smaller epsilon buys accuracy with sends: that is the knob a GTM team actually tunes.",
    ]
    return "\n".join(lines) + "\n"
