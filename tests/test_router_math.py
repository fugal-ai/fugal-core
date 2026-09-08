# Fugal — Apache-2.0. See NOTICE.
"""Unit tests for the routing formula and price calculation.

    python -m unittest discover -s tests -v

No network, no API key, no backbone: these test the pure math that turns a hidden
state into a model choice — sigmoid, L2 normalization, utility ranking, and the
price lookup that feeds spend caps. The head is synthesized inline with known values
so every expected output can be computed by hand.
"""
import json
import os
import tempfile
import unittest

import numpy as np

from fugal.router import load_head, load_prices


def _make_head(tmp, n=3, dim=4, lam=2.0, v2=False, context=None):
    """Write a tiny synthetic head and price sheet for testing.

    v2=True writes mean_in_tokens/mean_out_tokens (docs/HEAD_FORMAT.md) instead of a
    baked-in mean_cost, so load_head computes cost from the price sheet."""
    rng = np.random.default_rng(42)
    models = [f"test/model-{i}" for i in range(n)]
    W = rng.standard_normal((n, dim)).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    mean_cost = np.array([0.001, 0.005, 0.010], dtype=np.float64)[:n]
    head_path = os.path.join(tmp, "head.npz")
    arrays = {"W": W, "b": b, "models": np.array(models), "lam": np.float64(lam)}
    if v2:
        arrays["mean_in_tokens"] = np.array([100.0, 200.0, 300.0])[:n]
        arrays["mean_out_tokens"] = np.array([400.0, 500.0, 600.0])[:n]
    else:
        arrays["mean_cost"] = mean_cost
    if context is not None:
        arrays["context"] = np.array(context)
    np.savez(head_path, **arrays)
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


class TestLoadHead(unittest.TestCase):
    """load_head: both head formats, unpriced refusal, lambda resolution, subsetting."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._lam_env = os.environ.pop("FUGAL_LAMBDA", None)

    def tearDown(self):
        if self._lam_env is not None:
            os.environ["FUGAL_LAMBDA"] = self._lam_env

    def test_v1_uses_the_baked_in_mean_cost(self):
        head, prices_path, W, b, mc, models = _make_head(self.tmp)
        out = load_head(head, load_prices(prices_path)[0])
        self.assertEqual(out[0], models)
        np.testing.assert_array_equal(out[3], mc)
        self.assertEqual(out[4], 2.0)
        self.assertEqual(out[5], "standalone")     # absent context = standalone

    def test_v2_cost_comes_from_the_current_price_sheet(self):
        # THE v2 point: mean_cost = mean_in_tokens*price_in + mean_out_tokens*price_out,
        # evaluated against the sheet AS IT IS NOW, not as it was at fit time.
        head, prices_path, *_ , models = _make_head(self.tmp, v2=True)
        out = load_head(head, load_prices(prices_path)[0])
        # model i: in=(i+1)*0.5 $/M, out=(i+1)*1.5 $/M; tokens 100i+100 in, 100i+400 out
        expect = [(100 * 0.5 + 400 * 1.5) / 1e6,
                  (200 * 1.0 + 500 * 3.0) / 1e6,
                  (300 * 1.5 + 600 * 4.5) / 1e6]
        np.testing.assert_allclose(out[3], expect)

    def test_v2_cost_moves_when_prices_move(self):
        head, prices_path, *_ , models = _make_head(self.tmp, v2=True)
        doubled = {m: (2 * pin, 2 * pout)
                   for m, (pin, pout) in load_prices(prices_path)[0].items()}
        np.testing.assert_allclose(load_head(head, doubled)[3],
                                   2 * load_head(head, load_prices(prices_path)[0])[3])

    def test_context_flag_is_read(self):
        head, prices_path, *_ = _make_head(self.tmp, context="multiturn")
        self.assertEqual(load_head(head, load_prices(prices_path)[0])[5], "multiturn")

    def test_unpriced_model_is_a_hard_error(self):
        # Inventing a price would silently corrupt cost reports and spend caps.
        head, prices_path, *_ , models = _make_head(self.tmp)
        prices = load_prices(prices_path)[0]
        del prices[models[1]]
        with self.assertRaises(SystemExit):
            load_head(head, prices)
        # ...but excluding the unpriced model with a subset is fine.
        out = load_head(head, prices, models=[models[0], models[2]])
        self.assertEqual(out[0], [models[0], models[2]])

    def test_lambda_resolution_order(self):
        head, prices_path, *_ = _make_head(self.tmp, lam=2.0)
        prices = load_prices(prices_path)[0]
        self.assertEqual(load_head(head, prices)[4], 2.0)              # head default
        os.environ["FUGAL_LAMBDA"] = "5.5"
        try:
            self.assertEqual(load_head(head, prices)[4], 5.5)          # env beats head
            self.assertEqual(load_head(head, prices,
                                       router_lambda=9.0)[4], 9.0)     # arg beats env
        finally:
            del os.environ["FUGAL_LAMBDA"]

    def test_unknown_subset_model_is_refused(self):
        head, prices_path, *_ = _make_head(self.tmp)
        with self.assertRaises(SystemExit):
            load_head(head, load_prices(prices_path)[0], models=["not/in-head"])


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
