#!/usr/bin/env python3
# Fugal — Apache-2.0. See NOTICE.
"""
fetch_backbone.py — download the one thing this repo does not ship: the backbone.

    python scripts/fetch_backbone.py

Fetches Qwen/Qwen3-0.6B (Apache-2.0, ~1.5 GB) from HuggingFace into artifacts/Qwen3-0.6B,
which is where FUGAL_MODEL points by default — so after this, everything just runs.

The download is PINNED to one HuggingFace revision. The head was fit on hidden states from
one exact set of weights; if the upstream repo moved, a fresh download would route worse
with no error. A head that declares `backbone_revision` (docs/HEAD_FORMAT.md) wins over the
default below, so a newer head can bring its own backbone pin.

Already have Qwen3-0.6B on disk? Skip this and set FUGAL_MODEL to that directory instead —
and make sure it is the same revision.

Nothing fetched here is committed to the repo; see NOTICE. The router head
(data/router_head.npz, 73 KB) is original work and ships with the repo — it is the only
other thing the router needs.
"""
from __future__ import annotations
import os, sys

MODEL_ID = "Qwen/Qwen3-0.6B"
# The revision the shipped head was fit on and verified against (README tables). Overridden
# by the head's own `backbone_revision` when it declares one.
DEFAULT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "artifacts")
DEST = os.path.join(ART, "Qwen3-0.6B")
HEAD = os.environ.get("FUGAL_HEAD") or os.path.join(REPO, "data", "router_head.npz")
# What a usable checkout must contain. Checked afterwards rather than assumed: the original
# failure mode of this script was printing "Done" over an incomplete download and letting the
# router die with a confusing error several commands later.
REQUIRED = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]


def revision():
    """The head's declared backbone revision, else the default pin."""
    try:
        import numpy as np
        z = np.load(HEAD, allow_pickle=False)
        if "backbone_revision" in z.files and str(z["backbone_revision"]):
            return str(z["backbone_revision"]), "declared by " + os.path.relpath(HEAD, REPO)
    except (OSError, ValueError):
        pass
    return DEFAULT_REVISION, "default pin"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install -e .   (huggingface_hub is needed to fetch)")

    rev, why = revision()
    os.makedirs(ART, exist_ok=True)
    print(f"fetching {MODEL_ID} @ {rev[:12]} ({why}; Apache-2.0, ~1.5 GB) -> {DEST}")
    snapshot_download(MODEL_ID, revision=rev, local_dir=DEST)

    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(DEST, f))]
    if missing:
        sys.exit(f"\nFATAL: {DEST} is missing {', '.join(missing)}. The download did not "
                 f"complete; re-run this script.")

    print(f"\nDone. Backbone at {DEST}")
    print("Try it (no API key, no spend):")
    print('  python -m fugal --route "what is 15% of 240?"')


if __name__ == "__main__":
    main()
