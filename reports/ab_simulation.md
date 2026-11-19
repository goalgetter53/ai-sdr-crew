# Bayesian A/B module: simulation report

All numbers below are produced by `sdr-crew ab simulate` (seeded, reproducible). They validate the statistics code on synthetic data; they are not campaign results.

## 1. Calibration of P(variant beats control) (10000 simulated tests)

| predicted bin | n | mean predicted | observed frequency |
|---|---|---|---|
| 0.0-0.1 | 3936 | 0.010 | 0.007 |
| 0.1-0.2 | 345 | 0.148 | 0.174 |
| 0.2-0.3 | 252 | 0.248 | 0.198 |
| 0.3-0.4 | 254 | 0.348 | 0.331 |
| 0.4-0.5 | 232 | 0.453 | 0.440 |
| 0.5-0.6 | 208 | 0.549 | 0.524 |
| 0.6-0.7 | 214 | 0.652 | 0.603 |
| 0.7-0.8 | 279 | 0.751 | 0.774 |
| 0.8-0.9 | 362 | 0.854 | 0.834 |
| 0.9-1.0 | 3918 | 0.991 | 0.992 |

Max |predicted - observed| over bins with n >= 50: **0.049**; Brier score 0.0465.

## 2. A/A false-positive rate (1000 simulations, true rate 4.0% both arms, up to 3000 sends/arm, 20 looks)

| decision rule | false-positive rate |
|---|---|
| P(better) > 0.95, single look at the end | **5.9%** |
| P(better) > 0.95, checked at every look (peeking) | **20.8%** |

The expected-loss rule stopped in 95.0% of A/A runs (mean 914 sends/arm). In an A/A test either choice costs nothing, which is exactly what the loss rule optimises: it bounds the *cost* of a wrong call to epsilon instead of pretending to control a false-positive rate under peeking.

## 3. Decision quality with a real lift (4.0% -> 5.0%, 1000 simulations per row, cap 3000 sends/arm)

| epsilon (reply-rate points) | stopped | picked better arm | mean sends/arm at stop | mean regret (points) |
|---|---|---|---|---|
| 0.20 | 100.0% | 82.9% | 510 | 0.171 |
| 0.10 | 99.2% | 88.4% | 752 | 0.116 |
| 0.05 | 93.0% | 93.2% | 990 | 0.068 |

Smaller epsilon buys accuracy with sends: that is the knob a GTM team actually tunes.
