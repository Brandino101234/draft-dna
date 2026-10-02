"""Scoring rules for distributional predictions given as quantile grids.

Every model outputs predicted quantiles on the same grid `QS`. From those:
- pinball loss scores one quantile (lower is better; it penalizes a 90th-percentile
  prediction 9x more for being too low than too high, which is what a ceiling should do)
- CRPS (continuous ranked probability score) scores the whole distribution; on a
  quantile grid it is approximated by twice the average pinball loss
- tier probabilities come from the implied CDF, scored with the Brier score
"""

from __future__ import annotations

import numpy as np

QS = np.round(np.arange(0.05, 0.951, 0.05), 2)  # 19 quantiles
FLOOR, MEDIAN, CEILING = 0.25, 0.5, 0.9


def qidx(tau: float) -> int:
    return int(np.argmin(np.abs(QS - tau)))


def monotone(qpred: np.ndarray) -> np.ndarray:
    """Sort each row so predicted quantiles never cross."""
    return np.sort(qpred, axis=1)


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    diff = y - q
    return np.maximum(tau * diff, (tau - 1) * diff)


def crps(y: np.ndarray, qpred: np.ndarray) -> np.ndarray:
    """Per-observation CRPS approximated from the quantile grid."""
    losses = np.stack([pinball(y, qpred[:, i], t) for i, t in enumerate(QS)], axis=1)
    return 2 * losses.mean(axis=1)


def cdf_at(qpred: np.ndarray, t: float) -> np.ndarray:
    """P(Y <= t) for each row, by inverting the quantile grid.

    Linear between grid points; the tails beyond the 5th/95th percentiles are linear to
    0 and 1 over one extra grid step. Flat stretches (point masses, e.g. many players at
    exactly zero) resolve to the highest quantile level at that value.
    """
    out = np.empty(len(qpred))
    taus = np.concatenate([[0.0], QS, [1.0]])
    for i, q in enumerate(monotone(qpred)):
        lo = q[0] - (q[1] - q[0]) - 1e-9
        hi = q[-1] + (q[-1] - q[-2]) + 1e-9
        x = np.concatenate([[lo], q, [hi]])
        # For ties take the largest tau (right-continuous CDF).
        idx = np.searchsorted(x, t, side="right")
        if idx == 0:
            out[i] = 0.0
        elif idx >= len(x):
            out[i] = 1.0
        else:
            x0, x1 = x[idx - 1], x[idx]
            t0, t1 = taus[idx - 1], taus[idx]
            out[i] = t0 + (t1 - t0) * (t - x0) / (x1 - x0) if x1 > x0 else t1
    return np.clip(out, 0.0, 1.0)


def brier(prob: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    return (prob - outcome.astype(float)) ** 2


def coverage_below(y: np.ndarray, q: np.ndarray) -> float:
    """Share of outcomes below a predicted quantile (should equal its level).

    Exact ties count half: ~20% of players finish at exactly 0 (never played), and a
    floor of exactly 0 would otherwise look badly miscalibrated either way.
    """
    return float(((y < q) + 0.5 * (y == q)).mean())
