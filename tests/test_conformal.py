"""
Unit tests for conformal.py — verify the calibration math on synthetic
data before pointing it at depth.

Run: python -m pytest tests/test_conformal.py -v
"""

from __future__ import annotations

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conformal import (  # noqa: E402
    fit, interval, empirical_coverage, mean_interval_width,
    auroc_sigma_vs_error, evaluate,
)


def _gaussian_residual_data(n: int, seed: int = 0):
    """Generate (pred, sigma, gt) where residual ~ N(0, sigma).

    Conformal should produce empirical coverage close to 1 - alpha
    when the assumed score (|err|/sigma) is exchangeable between cal
    and test, which it is here by construction.
    """
    rng = np.random.default_rng(seed)
    pred = rng.normal(size=n) * 10.0 + 50.0
    sigma = np.abs(rng.normal(size=n)) * 2.0 + 0.5
    gt = pred + rng.normal(size=n) * sigma
    return pred, sigma, gt


def test_coverage_matches_target():
    # Large sample so the empirical coverage is tight to the target.
    pred, sigma, gt = _gaussian_residual_data(20_000, seed=1)
    n = pred.size // 2
    cal_pred, test_pred = pred[:n], pred[n:]
    cal_sigma, test_sigma = sigma[:n], sigma[n:]
    cal_gt, test_gt = gt[:n], gt[n:]

    for alpha in (0.05, 0.1, 0.2):
        rep = evaluate(cal_pred, cal_sigma, cal_gt,
                       test_pred, test_sigma, test_gt,
                       alpha=alpha)
        # Conformal guarantee: P(cover) >= 1 - alpha; tail allows some
        # slack but at 10k samples we expect within ~1%.
        assert rep.coverage >= 1 - alpha - 0.02, str(rep)
        assert rep.coverage <= 1 - alpha + 0.05, str(rep)


def test_auroc_perfect_when_sigma_equals_error():
    # If sigma is monotonically aligned with |err|, AUROC -> 1.
    rng = np.random.default_rng(0)
    err = np.abs(rng.normal(size=2000))
    sigma = err + 1e-6  # exactly aligned
    auc = auroc_sigma_vs_error(sigma, err)
    assert auc > 0.95, auc


def test_auroc_random_when_sigma_independent():
    rng = np.random.default_rng(0)
    err = np.abs(rng.normal(size=5000))
    sigma = rng.uniform(size=5000)
    auc = auroc_sigma_vs_error(sigma, err)
    assert 0.45 < auc < 0.55, auc


def test_fit_alpha_validation():
    s = np.array([0.1, 0.2, 0.3])
    for bad in (-0.1, 0.0, 1.0, 1.5):
        try:
            fit(s, bad)
        except ValueError:
            continue
        raise AssertionError(f"alpha={bad} should have raised")


def test_interval_shape():
    pred = np.arange(10).astype(float)
    sigma = np.ones(10)
    lo, hi = interval(pred, sigma, q=2.0)
    assert lo.shape == pred.shape
    assert np.allclose(hi - lo, 4.0)
    assert empirical_coverage(lo, hi, pred) == 1.0


def test_mean_width_finite():
    lo = np.array([0.0, 1.0, np.nan])
    hi = np.array([2.0, 4.0, 5.0])
    w = mean_interval_width(lo, hi)
    assert np.isfinite(w)
    assert abs(w - 2.5) < 1e-9


if __name__ == "__main__":
    test_coverage_matches_target()
    test_auroc_perfect_when_sigma_equals_error()
    test_auroc_random_when_sigma_independent()
    test_fit_alpha_validation()
    test_interval_shape()
    test_mean_width_finite()
    print("all tests pass")
