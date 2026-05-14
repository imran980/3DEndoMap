"""
Split conformal prediction for depth uncertainty.

Calibrates a scalar quantile q on a held-out set so that, at test
time, the interval `pred ± q * sigma` covers the true depth with
marginal probability >= 1 - alpha. Distribution-free, finite-sample,
assumes exchangeability between cal and test (which is why the eval
script splits at the sequence level, not the frame level).

This is the math module — no I/O, no torch, no plotting dependencies
beyond numpy + (optional) matplotlib for the calibration curve.
Tested via tests/test_conformal.py on synthetic data so we know the
coverage is exactly what it claims before pointing it at depth.

Reference: Vovk et al., "Algorithmic Learning in a Random World"
(2005); Lei et al. (JASA 2018) for split conformal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def fit(scores: np.ndarray, alpha: float) -> float:
    """Return the conformal quantile q on calibration scores.

    For split conformal at miscoverage alpha, q is the
    ceil((n+1)(1-alpha))/n empirical quantile (the finite-sample
    correction; converges to the (1-alpha)-quantile as n -> inf).

    scores: 1-D array of non-negative calibration scores s_i =
            |pred_i - gt_i| / sigma_i (any positive aggregation rule
            works as long as the same rule is used at test time).
    alpha:  miscoverage target. coverage = 1 - alpha.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    n = s.size
    if n == 0:
        raise ValueError("fit() received zero finite calibration scores.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
    # ceil((n+1)(1-alpha)) / n is the level; quantile() with that level
    # gives the conservative upper-quantile. Clip to 1.0 for tiny n.
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def interval(pred: np.ndarray, sigma: np.ndarray, q: float
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (lo, hi) for `pred +/- q * sigma`, same shape as pred."""
    half = q * np.asarray(sigma, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    return p - half, p + half


def empirical_coverage(lo: np.ndarray, hi: np.ndarray,
                       gt: np.ndarray) -> float:
    """Fraction of GT values inside [lo, hi]."""
    lo = np.asarray(lo).reshape(-1)
    hi = np.asarray(hi).reshape(-1)
    g = np.asarray(gt).reshape(-1)
    mask = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(g)
    if not mask.any():
        return float("nan")
    return float(((g[mask] >= lo[mask]) & (g[mask] <= hi[mask])).mean())


def mean_interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """Average |hi - lo| (a measure of how informative the intervals are
    at a given coverage)."""
    w = np.asarray(hi).reshape(-1) - np.asarray(lo).reshape(-1)
    w = w[np.isfinite(w)]
    return float(w.mean()) if w.size else float("nan")


def auroc_sigma_vs_error(sigma: np.ndarray, abs_error: np.ndarray
                         ) -> float:
    """AUROC of the binary task "is this point in the top half of
    abs_error?" using sigma as the score.

    Tells us whether sigma is even rank-correlated with true error —
    a useful sanity check before trusting the conformal interval
    width to mean anything. Random sigma -> 0.5; perfect -> 1.0.
    """
    s = np.asarray(sigma, dtype=np.float64).reshape(-1)
    e = np.asarray(abs_error, dtype=np.float64).reshape(-1)
    m = np.isfinite(s) & np.isfinite(e)
    s, e = s[m], e[m]
    if s.size < 2:
        return float("nan")
    median_e = np.median(e)
    y = (e >= median_e).astype(np.int32)
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    # Mann-Whitney U / |y| / |~y| = AUROC. Use rankdata to avoid an
    # O(n^2) implementation.
    from scipy.stats import rankdata
    ranks = rankdata(s)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    sum_ranks_pos = float(ranks[y == 1].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def calibration_curve(scores: np.ndarray,
                      alphas: Optional[np.ndarray] = None
                      ) -> Dict[str, np.ndarray]:
    """Predicted-vs-observed coverage across a sweep of alphas.

    For each target alpha, fit q on the same scores (treating them as
    cal) and report predicted coverage = 1 - alpha vs. observed
    coverage on the same scores under the test inclusion rule s <= q.
    On the cal set this gives the empirical coverage we'd report if
    we used these scores as a test set — useful diagnostic, not a
    real held-out evaluation.
    """
    if alphas is None:
        alphas = np.linspace(0.01, 0.20, 20)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    s = s[np.isfinite(s)]
    predicted = 1.0 - alphas
    observed = np.array(
        [float((s <= fit(s, a)).mean()) for a in alphas]
    )
    return dict(alpha=alphas, predicted=predicted, observed=observed)


@dataclass
class EvalReport:
    """A bundle of conformal eval metrics, easy to pickle / print."""
    n_cal: int
    n_test: int
    alpha: float
    q: float
    coverage: float
    mean_width: float
    auroc: float

    def __str__(self) -> str:
        return (
            f"[conformal] alpha={self.alpha:.3f}  "
            f"q={self.q:.4f}  coverage={self.coverage:.4f} "
            f"(target {1 - self.alpha:.4f})  "
            f"width={self.mean_width:.2f}  AUROC(sigma,|err|)={self.auroc:.3f}  "
            f"n_cal={self.n_cal}  n_test={self.n_test}"
        )


def evaluate(cal_pred: np.ndarray, cal_sigma: np.ndarray, cal_gt: np.ndarray,
             test_pred: np.ndarray, test_sigma: np.ndarray, test_gt: np.ndarray,
             alpha: float) -> EvalReport:
    """End-to-end split-conformal eval. cal_* are used to fit q;
    test_* are used to measure coverage and width.

    Inputs are flat numpy arrays of matched length per split. The
    eval script (eval_conformal_pixel.py) handles the
    frame/pixel/sequence subsampling that produces them.
    """
    cs = np.abs(np.asarray(cal_pred) - np.asarray(cal_gt)) \
        / np.clip(np.asarray(cal_sigma), 1e-9, None)
    q = fit(cs, alpha)
    lo, hi = interval(np.asarray(test_pred), np.asarray(test_sigma), q)
    cov = empirical_coverage(lo, hi, np.asarray(test_gt))
    width = mean_interval_width(lo, hi)
    auc = auroc_sigma_vs_error(
        np.asarray(test_sigma),
        np.abs(np.asarray(test_pred) - np.asarray(test_gt)),
    )
    return EvalReport(
        n_cal=int(np.isfinite(cs).sum()),
        n_test=int(np.isfinite(np.asarray(test_gt)).sum()),
        alpha=float(alpha),
        q=float(q),
        coverage=float(cov),
        mean_width=float(width),
        auroc=float(auc),
    )
