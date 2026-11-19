# Regression report: v1 -> v2 [MOCK MODE]

Golden set: 22 fictional accounts. Numbers come from deterministic offline runs of the mock model; they measure the harness, guardrails and prompt-rule adherence of the mock, not real model quality. Token and cost figures are estimates (~4 chars/token, assumed Sonnet-class prices).

| metric | v1 | v2 | delta | verdict |
|---|---|---|---|---|
| success_rate | 1 | 1 | +0 | same |
| status_accuracy | 1 | 1 | +0 | same |
| angle_accuracy | 1 | 1 | +0 | same |
| first_pass_clean_rate | 0.1667 | 1 | +0.8333 | better |
| mean_revisions | 1 | 0 | -1 | better |
| citation_coverage | 1 | 1 | +0 | same |
| final_lint_errors | 0 | 0 | +0 | same |
| ungrounded_facts_dropped | 3 | 0 | -3 | better |
| total_tokens | 209053 | 192580 | -1.647e+04 | better |
| cost_usd_est | 0.6986 | 0.4205 | -0.2781 | better |
| cost_per_ready_sequence_usd_est | 0.0388 | 0.0234 | -0.0154 | better |
| cache_savings_pct | 21.1 | 44.14 | +23.04 | better |
| latency_p50_ms | 12.06 | 8.235 | -3.825 | same |
| judge_relevance | 5 | 5 | +0 | same |
| judge_personalisation | 4.944 | 4 | -0.944 | WORSE |
| judge_clarity | 5 | 5 | +0 | same |
| judge_cta | 5 | 5 | +0 | same |
| judge_tone | 5 | 5 | +0 | same |
| judge_mean | 4.989 | 4.8 | -0.189 | WORSE |

## Per-account regressions

- brightforge.example: judge 5.0 -> 4.8 (personalisation 5->4)
- copperline.example: judge 5.0 -> 4.8 (personalisation 5->4)
- fernway.example: judge 5.0 -> 4.8 (personalisation 5->4)
- harborlight.example: judge 5.0 -> 4.8 (personalisation 5->4)
- northpeak.example: judge 5.0 -> 4.8 (personalisation 5->4)
- quarrystone.example: judge 5.0 -> 4.8 (personalisation 5->4)
- bluefin.example: judge 5.0 -> 4.8 (personalisation 5->4)
- oakridge.example: judge 5.0 -> 4.8 (personalisation 5->4)
- pinecrest.example: judge 5.0 -> 4.8 (personalisation 5->4)
- saltmarsh.example: judge 5.0 -> 4.8 (personalisation 5->4)
- tidewater.example: judge 5.0 -> 4.8 (personalisation 5->4)
- umberto.example: judge 5.0 -> 4.8 (personalisation 5->4)
- vantageclinics.example: judge 5.0 -> 4.8 (personalisation 5->4)
- willowbank.example: judge 5.0 -> 4.8 (personalisation 5->4)
- xenith.example: judge 5.0 -> 4.8 (personalisation 5->4)
- yellowpine.example: judge 5.0 -> 4.8 (personalisation 5->4)
- zephyr.example: judge 5.0 -> 4.8 (personalisation 5->4)

## Release gate

- FAIL judge_mean: 4.989 -> 4.8
