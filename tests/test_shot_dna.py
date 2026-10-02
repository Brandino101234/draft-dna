import numpy as np
import pytest

from draft_dna.features.shot_dna import BetaPrior, fit_beta_prior


def test_beta_prior_recovers_true_talent_spread() -> None:
    rng = np.random.default_rng(0)
    true_mean, true_sd = 0.35, 0.04  # D-I three-point shooting
    talent = rng.normal(true_mean, true_sd, 4000).clip(0.05, 0.95)
    att = rng.integers(50, 300, 4000)
    made = rng.binomial(att, talent)
    prior = fit_beta_prior(made, att)
    expected_kappa = true_mean * (1 - true_mean) / true_sd**2 - 1  # ~141
    assert prior.mean == pytest.approx(true_mean, abs=0.005)
    assert prior.kappa == pytest.approx(expected_kappa, rel=0.25)


def test_shrinkage_pulls_small_samples_harder() -> None:
    prior = BetaPrior(mean=0.35, kappa=140)
    hot_small = prior.shrink(np.array([9]), np.array([10]))[0]  # 90% on 10 shots
    hot_large = prior.shrink(np.array([180]), np.array([400]))[0]  # 45% on 400 shots
    assert 0.35 < hot_small < 0.40  # 10 shots barely move the estimate
    assert 0.41 < hot_large < 0.45  # 400 shots mostly kept
