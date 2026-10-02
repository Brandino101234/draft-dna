"""Bayesian grade updating: blend the draft-night projection with observed NBA play.

Everything is on a square-root scale, s(v) = sqrt(max(v, 0)), where peak value is
roughly normal and "no NBA value" sits at 0.

- Prior: the draft-night projection of the player's peak through year 8, summarized as
  Normal(m0, s0) on the sqrt scale (mean and spread of the projected quantile grid).
- Likelihood: from history, after N seasons the observed peak so far relates to the
  eventual year-8 peak as   s(peak_N) = a_N + b_N * s(peak_8) + noise,  noise ~ N(0, sigma_N).
  b_N grows toward 1 and sigma_N shrinks as seasons accumulate, so later seasons say more.
- Posterior (standard linear-Gaussian update):
      precision = 1/s0^2 + b_N^2/sigma_N^2
      mean      = (m0/s0^2 + b_N (obs - a_N)/sigma_N^2) / precision
  The share of the posterior driven by data is (b_N^2/sigma_N^2) / precision; it is
  learned from history, not set by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from draft_dna.eval.metrics import QS

FINAL_N = 8


def s(v: np.ndarray | pd.Series | float) -> np.ndarray:
    return np.sqrt(np.maximum(np.asarray(v, dtype=float), 0.0))


@dataclass(frozen=True)
class Measurement:
    n: int
    a: float
    b: float
    sigma: float

    @property
    def information(self) -> float:
        return self.b**2 / self.sigma**2


def fit_measurement(peaks: pd.DataFrame, ns: range = range(1, FINAL_N)) -> dict[int, Measurement]:
    """peaks: one row per player with columns 1..8 (peak value through N seasons)."""
    d = peaks.dropna(subset=[FINAL_N])
    out = {}
    final = s(d[FINAL_N])
    for n in ns:
        obs = s(d[n])
        b, a = np.polyfit(final, obs, 1)
        sigma = float(np.std(obs - (a + b * final), ddof=2))
        out[n] = Measurement(n, float(a), float(b), max(sigma, 1e-3))
    return out


def prior_from_grid(qgrid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd on the sqrt scale of each row's projected quantile grid (the grid is
    evenly spaced in probability, so it approximates the distribution)."""
    t = s(qgrid)
    return t.mean(axis=1), np.maximum(t.std(axis=1), 0.05)


@dataclass(frozen=True)
class Posterior:
    mean: np.ndarray
    sd: np.ndarray
    data_weight: np.ndarray

    def quantiles(self, levels: np.ndarray = QS, z: np.ndarray | None = None) -> np.ndarray:
        """Quantiles of the year-8 peak. `z` overrides the normal percentiles with
        calibrated ones (see `calibrated_z`)."""
        zz = (norm.ppf(levels) if z is None else z)[None, :]
        return np.maximum(self.mean[:, None] + zz * self.sd[:, None], 0.0) ** 2


def update(
    m0: np.ndarray, s0: np.ndarray, obs_peak: np.ndarray, meas: Measurement | None
) -> Posterior:
    """Posterior for the year-8 peak after observing peak-so-far (None = no seasons yet)."""
    m0, s0 = np.asarray(m0, float), np.asarray(s0, float)
    if meas is None:
        return Posterior(m0, s0, np.zeros(len(m0)))
    prior_prec = 1 / s0**2
    data_prec = meas.information
    prec = prior_prec + data_prec
    mean = (m0 * prior_prec + meas.b * (s(obs_peak) - meas.a) / meas.sigma**2) / prec
    return Posterior(mean, np.sqrt(1 / prec), data_prec / prec)


def grade_letter(posterior_median: np.ndarray, prior_grid: np.ndarray) -> np.ndarray:
    """Where the current expected outcome sits inside the draft-night range (PIT, with the
    midpoint rule for piles at exactly 0):
    A: above the projected ceiling (90th pct); B: above the median; C: between the floor
    (25th) and the median; D: below the floor."""
    from draft_dna.eval.phase6 import pit

    pct = pit(prior_grid, posterior_median)
    return np.select([pct >= 0.9, pct >= 0.5, pct >= 0.25], ["A", "B", "C"], "D")


def status(n_seasons: int, retired: bool) -> str:
    if retired and n_seasons > 0:
        return "Career Grade (retired)"
    if n_seasons == 0:
        return "Projection"
    if n_seasons == 1:
        return "Provisional (low confidence)"
    if n_seasons <= 3:
        return "Provisional (medium confidence)"
    if n_seasons < FINAL_N:
        return "Year-4 Verdict"
    return "Career Grade"


N_BINS = 5


@dataclass(frozen=True)
class Calibration:
    """Empirical error quantiles (sqrt scale) by bin of posterior mean, for one N.

    A normal shape cannot represent the pile of careers ending at exactly 0, and how big
    that pile is depends on the player (a late second-rounder vs a top-5 pick). So errors
    (actual - posterior mean) are measured separately for low/middle/high projected
    players on fitting data, and their empirical quantiles replace the normal ones: a
    conformal-style correction that keeps the posterior mean.
    """

    edges: np.ndarray  # inner bin edges on the posterior-mean scale
    resid_q: np.ndarray  # (bins, levels)

    def quantiles(self, post: Posterior) -> np.ndarray:
        b = np.searchsorted(self.edges, post.mean, side="right")
        return np.maximum(post.mean[:, None] + self.resid_q[b], 0.0) ** 2


def fit_calibration(post: Posterior, final: np.ndarray, levels: np.ndarray = QS) -> Calibration:
    edges = np.quantile(post.mean, np.linspace(0, 1, N_BINS + 1)[1:-1])
    b = np.searchsorted(edges, post.mean, side="right")
    resid = s(final) - post.mean
    rq = np.vstack([np.quantile(resid[b == k], levels) for k in range(N_BINS)])
    return Calibration(edges, rq)
