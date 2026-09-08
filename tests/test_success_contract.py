"""Success semantics and strict artifact boundary; legacy behavior has its own suite."""
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from fugal import router
from fugal import success_contract as c


def fixture():
    manifest = {"version": "fugal-token-statistics-v1", "profile_id": c.PROFILE_ID,
                "status": "synthetic-test-only", "sources": ["fixture"], "worker_profile": "mock",
                "models": [{"id": m, "samples": 2, "mean_in_tokens": 100., "mean_out_tokens": 50.}
                           for m in ("first", "second", "third")]}
    z = c.make_head(np.zeros((3, 1024)), np.array([0., .1, .2]), ["first", "second", "third"], manifest, "test-only")
    return z, manifest


def serialized(z):
    buf = io.BytesIO()
    np.savez(buf, **z)
    return buf.getvalue()


class SuccessContractTests(unittest.TestCase):
    def test_reference_conformance_fixtures(self):
        fixture_path = Path(__file__).resolve().parents[1] / "data/success_conformance.json"
        for case in json.loads(fixture_path.read_text())["cases"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(c.rank(c.sigmoid(case["logits"]), case["costs"], case["lam"]).tolist(), case["order"])

    def test_sigmoid_saturation_and_independence(self):
        with np.errstate(over="raise"):
            np.testing.assert_array_equal(c.sigmoid(np.array([-1e300, 0., 1e300])), [0, .5, 1])
        p = c.predictions(np.ones((3, 1024)), np.arange(3), np.ones(1024))
        for lam in (0, .5, 1, 2, 5):
            c.rank(p, [.001, .002, .003], lam)
            np.testing.assert_array_equal(p, c.predictions(np.ones((3, 1024)), np.arange(3), np.ones(1024)))

    def test_grid_ties_and_price_changes(self):
        np.testing.assert_array_equal(c.rank([.50001, .50002], [0, 0]), [0, 1])
        self.assertEqual(c.rank([.5, .6], [0, .2], 0)[0], 1)
        self.assertEqual(c.rank([.5, .6], [0, .2], 1)[0], 0)
        self.assertEqual(c.rank([.5, .6], [0, .01], 1)[0], 1)
        for lam in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                c.rank([.5], [0], lam)

    def test_artifact_roundtrip_manifest_and_subsets_keep_row_order(self):
        z, manifest = fixture()
        c.check_manifest(c.load(serialized(z)), manifest)
        with tempfile.TemporaryDirectory() as d, mock.patch.dict("os.environ", {}, clear=True):
            path = Path(d) / "head.npz"
            path.write_bytes(serialized(z))
            head = router.load_head(path, {m: (1e-6, 2e-6) for m in z["models"]}, models=["third", "first"], router_lambda=2)
            self.assertEqual(head.models, ["first", "third"])
            self.assertEqual(head.fmt, c.CONTRACT)
            self.assertEqual(head.lam, 2)
            np.testing.assert_allclose(head.mean_cost, [.0002, .0002])
        bad = json.loads(json.dumps(manifest))
        bad["models"][0]["mean_in_tokens"] = 200
        with self.assertRaises(ValueError):
            c.check_manifest(z, bad)
        with self.assertRaises(ValueError):
            c.check_manifest(z, manifest, deployable=True)

    def test_invalid_artifacts_rejected(self):
        z, _ = fixture()
        for key, val in [("contract", "v1"), ("profile_id", "wrong"), ("backbone_revision", "wrong"),
                         ("W", np.zeros((3, 10))), ("b", np.array([0, np.inf, 1])),
                         ("models", ["a", "a", "b"]), ("mean_in_tokens", [-1, 0, 0]),
                         ("lam", float("nan")), ("provenance", "")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                c.load(serialized(dict(z, **{key: np.asarray(val)})))
        with self.assertRaises(ValueError):
            c.load(serialized(dict(z, unknown=np.array(1))))
        with self.assertRaises(ValueError):
            c.load(serialized(dict(z, W=np.array([{}], dtype=object))))
        with self.assertRaises(ValueError):
            c.read_archive(b"x" * (c.MAX_BYTES + 1))

    def test_missing_version_cannot_downgrade_to_legacy(self):
        z, _ = fixture()
        z.pop("contract")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "head.npz"
            path.write_bytes(serialized(z))
            with self.assertRaises(ValueError, msg="missing explicit contract"):
                router.load_head(path, {m: (0., 0.) for m in z["models"]})

    def test_cache_rejects_other_profile_or_questions(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.npz"
            c.save_cache(path, ["q"], np.ones((1, 1024)))
            self.assertEqual(c.load_cache(path, ["q"]).shape, (1, 1024))
            for questions, profile in ((["changed"], c.PROFILE_ID), (["q"], "other")):
                with self.assertRaises(ValueError):
                    c.load_cache(path, questions, profile)
            np.savez(path, H=np.ones((1, 1024)))
            with self.assertRaises(KeyError):
                c.load_cache(path, ["q"])

    def test_question_format(self):
        for q in ("", "short", "🙂 café 漢字", "a\nb", "long " * 4000):
            self.assertEqual(c.format_question(q), f"system: {router.ROUTER_SYSTEM_PROMPT}\nuser: {q}")

    def test_custom_backbone_directory_is_checked(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "config.json").write_text('{}')
            with self.assertRaises(ValueError):
                c.check_backbone(d)

    def test_success_route_once_preserves_full_worker_request(self):
        f = router.Fugal.__new__(router.Fugal)
        f.head_format, f.head_context = c.CONTRACT, "standalone"
        f.models, f.W, f.b = ["first", "second"], np.zeros((2, 1024)), np.zeros(2)
        f.mean_cost, f.lam = np.array([.001, .002]), 1
        f.max_out_tokens, f.prices = {}, {"first": (1e-6, 2e-6)}
        import threading
        f._rlock = threading.Lock()
        hidden = mock.Mock()
        hidden.float.return_value.cpu.return_value.numpy.return_value = np.zeros(1024)
        f.router = mock.Mock()
        f.router.hidden.return_value = hidden
        question, history = "word " * 4000, [{"role": "user", "content": "previous"}]
        with mock.patch.object(router, "or_call", return_value=("answer", {})) as worker:
            f.answer(question, history=history)
        self.assertEqual(worker.call_count, 1)
        self.assertEqual(worker.call_args.args[:2], ("first", question))
        self.assertEqual(worker.call_args.kwargs["history"], history)
        msgs = f.router.hidden.call_args.args[0]
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[-1]["content"], question)
