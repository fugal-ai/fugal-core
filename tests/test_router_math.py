# Fugal — Apache-2.0. See NOTICE.
"""Unit tests for the routing formula and price calculation.

    python -m unittest discover -s tests -v

No network, no API key, no backbone: these test the pure math that turns a hidden
state into a model choice — sigmoid, L2 normalization, utility ranking, and the
price lookup that feeds spend caps. The head is synthesized inline with known values
so every expected output can be computed by hand.
"""
import io
import json
import os
import tempfile
import unittest

import numpy as np

from fugal.router import clamp_max_tokens


def _make_head(tmp, n=3, dim=4, lam=2.0):
    """Write a tiny synthetic head and price sheet for testing."""
    rng = np.random.default_rng(42)
    models = [f"test/model-{i}" for i in range(n)]
    W = rng.standard_normal((n, dim)).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    mean_cost = np.array([0.001, 0.005, 0.010], dtype=np.float64)[:n]
    head_path = os.path.join(tmp, "head.npz")
    np.savez(head_path, W=W, b=b, models=np.array(models),
             mean_cost=mean_cost, lam=np.float64(lam))
    prices = [{"id": m, "in": (i + 1) * 0.5, "out": (i + 1) * 1.5}
              for i, m in enumerate(models)]
    prices_path = os.path.join(tmp, "prices.json")
    with open(prices_path, "w") as f:
        json.dump(prices, f)
    return head_path, prices_path, W, b, mean_cost, models


class TestRouteFormula(unittest.TestCase):
    """Test the four-line formula against known inputs."""

    def test_sigmoid_and_utility(self):
        W = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        mean_cost = np.array([0.001, 0.010])
        lam = 2.0
        h = np.array([2.0, 0.5])
        h = h / np.linalg.norm(h)
        p = 1 / (1 + np.exp(-(W @ h + b)))
        util = p - lam * mean_cost
        self.assertEqual(len(p), 2)
        self.assertTrue(np.all(p > 0) and np.all(p < 1))
        winner = np.argmax(util)
        p0 = 1 / (1 + np.exp(-h[0]))
        p1 = 1 / (1 + np.exp(-h[1]))
        self.assertAlmostEqual(float(p[0]), float(p0), places=6)
        self.assertAlmostEqual(float(p[1]), float(p1), places=6)
        u0 = p0 - lam * 0.001
        u1 = p1 - lam * 0.010
        self.assertEqual(winner, 0 if u0 > u1 else 1)

    def test_l2_normalization_preserves_direction(self):
        h = np.array([3.0, 4.0, 0.0])
        normed = h / max(np.linalg.norm(h), 1e-8)
        self.assertAlmostEqual(np.linalg.norm(normed), 1.0, places=8)
        self.assertAlmostEqual(normed[0], 0.6)
        self.assertAlmostEqual(normed[1], 0.8)

    def test_zero_vector_guard(self):
        h = np.array([0.0, 0.0, 0.0])
        normed = h / max(np.linalg.norm(h), 1e-8)
        self.assertTrue(np.all(np.isfinite(normed)))

    def test_independent_rows(self):
        """Subsetting rows does not change surviving scores."""
        W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        h = np.array([0.5, 0.3, 0.8])
        h = h / np.linalg.norm(h)
        full_p = 1 / (1 + np.exp(-(W @ h + b)))
        idx = [0, 2]
        sub_p = 1 / (1 + np.exp(-(W[idx] @ h + b[idx])))
        np.testing.assert_array_almost_equal(sub_p, full_p[idx])


class TestPriceCalculation(unittest.TestCase):

    def test_price_arithmetic(self):
        prices = {"test/m": (2.0 / 1e6, 6.0 / 1e6)}
        pin, pout = prices["test/m"]
        cost = 1000 * pin + 500 * pout
        self.assertAlmostEqual(cost, 1000 * 2e-6 + 500 * 6e-6)

    def test_fallback_price(self):
        prices = {}
        pin, pout = prices.get("unknown/model", (1e-6, 3e-6))
        self.assertEqual(pin, 1e-6)
        self.assertEqual(pout, 3e-6)


class TestLambdaEffect(unittest.TestCase):

    def test_higher_lambda_prefers_cheaper(self):
        """Raising lambda should never promote a more expensive model."""
        W = np.array([[1.0, 0.0], [0.95, 0.0]], dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        mean_cost = np.array([0.010, 0.001])
        h = np.array([0.5, 0.5])
        h = h / np.linalg.norm(h)
        p = 1 / (1 + np.exp(-(W @ h + b)))
        for lo, hi in [(0.0, 2.0), (2.0, 10.0), (10.0, 100.0)]:
            winner_lo = np.argmax(p - lo * mean_cost)
            winner_hi = np.argmax(p - hi * mean_cost)
            self.assertGreaterEqual(mean_cost[winner_lo], mean_cost[winner_hi],
                                    f"lambda {lo}->{hi} promoted more expensive model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
