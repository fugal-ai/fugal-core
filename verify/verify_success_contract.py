# Fugal — Apache-2.0. See NOTICE.
"""Verify local CPU embedding and routing conformance, optionally against subnet.

No worker calls. --subnet is an exact local checkout; output pins both commits.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fugal import success_contract as c  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", default=os.environ.get("FUGAL_MODEL", str(REPO / "artifacts/Qwen3-0.6B")))
    p.add_argument("--subnet")
    p.add_argument("--head", help="Subnet-exported head to load through FUGAL_HEAD")
    p.add_argument("--prices")
    p.add_argument("--output")
    args = p.parse_args()
    import torch
    torch.set_num_threads(4)
    from fugal.router import FugalRouter
    router = FugalRouter(args.backbone, success_profile=True)
    fixture = json.loads((REPO / "data/success_conformance.json").read_text())
    questions = [q if isinstance(q, str) else q["repeat"] * q["count"] for q in fixture["questions"]]
    single = np.stack([router.embed_questions([q])[0] for q in questions])
    batched = router.embed_questions(questions, batch_size=2)
    np.testing.assert_allclose(single, batched, atol=1e-5, rtol=1e-5)
    tokens = c.tokenize(router.tok, questions)
    assert tokens["input_ids"].shape[1] == 2048
    assert tokens["attention_mask"].sum(1).tolist() == fixture["token_lengths"]
    assert c.digest(tokens["input_ids"].tolist()) == fixture["token_ids_sha256"]
    results = {"core_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               "profile_id": c.PROFILE_ID, "atol": 1e-5, "rtol": 1e-5,
               "batch_single_max_abs": float(np.max(np.abs(single - batched))),
               "token_lengths": tokens["attention_mask"].sum(1).tolist(),
               "token_ids_sha256": c.digest(tokens["input_ids"].tolist()), "questions": len(questions)}
    rng = np.random.default_rng(17)
    W, b = rng.normal(size=(3, 1024)).astype(np.float32), rng.normal(size=3).astype(np.float32)
    costs = np.array([.001, .005, .02])
    first = c.rank(c.predictions(W, b, single), costs)
    batch_order = c.rank(c.predictions(W, b, batched), costs)
    np.testing.assert_array_equal(first, batch_order)
    if args.subnet:
        source = Path(args.subnet) / "fugal_subnet/vendor/success_contract.py"
        assert source.read_bytes() == (REPO / "fugal/success_contract.py").read_bytes()
        spec = importlib.util.spec_from_file_location("subnet_success", source)
        subnet = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(subnet)
        sub_tokens = subnet.tokenize(router.tok, questions)
        np.testing.assert_array_equal(tokens["input_ids"], sub_tokens["input_ids"])
        h = subnet.embed(router.tok, router.model.model, questions, batch_size=2)
        np.testing.assert_allclose(single, h, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(first, subnet.rank(subnet.predictions(W, b, h), costs))
        results["subnet_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.subnet, text=True).strip()
        results["vendor_sha256"] = c.file_hash(source)
    if args.head:
        if not args.prices:
            p.error("--head requires --prices")
        os.environ["FUGAL_HEAD"] = args.head
        os.environ["FUGAL_PRICES"] = args.prices
        os.environ["FUGAL_MODEL"] = args.backbone
        # FUGAL_HEAD is resolved at import in the serving module. Reload it so
        # this exercises the environment-driven serving interface exactly.
        import fugal.router as serving
        serving = importlib.reload(serving)
        # Release the conformance backbone before loading through serving.
        del router
        import gc
        gc.collect()
        f = serving.Fugal()
        z = c.load(Path(args.head).read_bytes())
        probabilities = c.predictions(z["W"], z["b"], single)
        expected = c.rank(probabilities, f.mean_cost, f.lam)
        selected = []
        for i, q in enumerate(questions):
            models, p_solve = f.route(q)
            assert models == [f.models[j] for j in expected[i]]
            np.testing.assert_allclose(p_solve, probabilities[i, expected[i]], atol=1e-12, rtol=1e-12)
            selected.append(models[0])
        results.update(exported_head_sha256=c.file_hash(args.head), FUGAL_HEAD_loaded=True, selections=selected)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
