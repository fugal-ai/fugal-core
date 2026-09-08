# Fugal — Apache-2.0. See NOTICE.
"""Portable success-head contract. NumPy only until an embedding is requested.

This file is vendored verbatim by subnet; no serving dependency is required.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np

CONTRACT = "fugal-success-v1"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_ID = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT = (
    "You are a routing model. Given a question, your hidden state will be used to "
    "predict which language model can best answer it. Read the question carefully.")
PROFILE = {
    "version": "qwen3-success-2048-v1", "model": MODEL_ID,
    "model_revision": REVISION, "tokenizer_revision": REVISION,
    "system_prompt": SYSTEM_PROMPT, "format": "system: {system}\nuser: {question}",
    "max_length": 2048, "truncation_side": "right", "padding_side": "right",
    "pooling": "attention-mask-mean-l2", "device": "cpu", "dtype": "float32",
}
# Hashes from the pinned HF snapshot, including tokenizer configuration. A directory
# name or a user-written revision marker does not establish the bytes being loaded.
BACKBONE_FILES = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}
QUANTUM = 1e-4
MAX_BYTES = 1024 * 1024
MAX_EXPANDED_BYTES = 8 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


PROFILE_ID = digest(PROFILE)


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_backbone(directory):
    directory = Path(directory)
    for name, expected in BACKBONE_FILES.items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"backbone/tokenizer identity mismatch: {name}")
    # Prevent transformers preferring another local weight format or custom code.
    permitted = set(BACKBONE_FILES) | {"generation_config.json"}
    for path in directory.iterdir():
        if path.suffix in {".safetensors", ".bin", ".py", ".json"} and path.name not in permitted:
            raise ValueError(f"unexpected model loading file: {path.name}")


def format_question(question):
    if not isinstance(question, str):
        raise ValueError("question must be a string")
    return f"system: {SYSTEM_PROMPT}\nuser: {question}"


def tokenize(tokenizer, questions, max_length=2048):
    if max_length not in (512, 2048):
        raise ValueError("unsupported input limit")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer([format_question(q) for q in questions], padding=True,
                     truncation=True, max_length=max_length, return_tensors="pt")


def embed(tokenizer, model, questions, batch_size=8, max_length=2048):
    """512 is an explicitly experimental variant, never the deployable profile."""
    import torch
    if next(model.parameters()).device.type != "cpu" or next(model.parameters()).dtype != torch.float32:
        raise ValueError("success embeddings require CPU float32")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    rows = []
    with torch.inference_mode():
        for start in range(0, len(questions), batch_size):
            inputs = tokenize(tokenizer, questions[start:start + batch_size], max_length)
            outputs = model(**inputs).last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            pooled = (outputs * mask).sum(1) / mask.sum(1).clamp_min(1)
            rows.append(torch.nn.functional.normalize(pooled, p=2, dim=1).cpu().numpy())
    return np.concatenate(rows) if rows else np.empty((0, 1024), dtype=np.float32)


def sigmoid(logits):
    x = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1 / (1 + np.exp(-x[positive]))
    exp = np.exp(x[~positive])
    out[~positive] = exp / (1 + exp)
    return out


def predictions(W, b, hidden):
    return sigmoid(np.asarray(hidden, dtype=np.float64) @ np.asarray(W, dtype=np.float64).T + b)


def rank(probabilities, costs, lam=1.0):
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("lambda must be finite and nonnegative")
    p, c = np.asarray(probabilities, dtype=np.float64), np.asarray(costs, dtype=np.float64)
    if not np.isfinite(p).all() or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("invalid predictions or costs")
    with np.errstate(over="raise", invalid="raise"):
        utility = np.round((p - lam * c) / QUANTUM) * QUANTUM
    return np.argsort(-utility, axis=-1, kind="stable")


def read_archive(data):
    """Bound archive allocation before loading. Never deserialize Python objects."""
    if len(data) > MAX_BYTES:
        raise ValueError("head exceeds archive size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
            names = [x.filename for x in entries]
            if len(entries) > 32 or len(set(names)) != len(names):
                raise ValueError("invalid archive entries")
            if sum(x.file_size for x in entries) > MAX_EXPANDED_BYTES:
                raise ValueError("head exceeds expanded size limit")
            if any(not n.endswith(".npy") or "/" in n for n in names):
                raise ValueError("invalid archive member")
            for entry in entries:
                with archive.open(entry) as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise ValueError("unsupported NPY version")
                    if dtype.hasobject or any(n < 0 for n in shape):
                        raise ValueError("invalid array dtype/shape")
                    size = math.prod(shape) * dtype.itemsize
                    if size > MAX_EXPANDED_BYTES or size != entry.file_size - member.tell():
                        raise ValueError("array shape exceeds archive payload")
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except (OSError, zipfile.BadZipFile) as e:
        raise ValueError("invalid head archive") from e


def scalar(z, key):
    a = z[key]
    if a.shape != () or a.dtype.kind != "U":
        raise ValueError(f"{key} must be a Unicode scalar")
    return str(a)


def validate_head(z):
    required = {"W", "b", "models", "contract", "profile_id", "backbone_revision",
                "provenance", "lam", "mean_in_tokens", "mean_out_tokens", "cost_profile_id"}
    if set(z) != required:
        raise ValueError(f"unsupported head fields: {set(z) ^ required}")
    for key, expected in (("contract", CONTRACT), ("profile_id", PROFILE_ID),
                          ("backbone_revision", REVISION)):
        if scalar(z, key) != expected:
            raise ValueError(f"unsupported {key}")
    for key in ("provenance", "cost_profile_id"):
        if not scalar(z, key).strip():
            raise ValueError(f"missing {key}")
    if len(scalar(z, "cost_profile_id")) != 64 or any(ch not in "0123456789abcdef" for ch in scalar(z, "cost_profile_id")):
        raise ValueError("invalid cost profile identity")
    models = z["models"]
    if models.ndim != 1 or models.dtype.kind != "U" or not 1 <= len(models) <= 64:
        raise ValueError("invalid models")
    if len(set(models.tolist())) != len(models) or any(not m.strip() or len(m) > 256 for m in models):
        raise ValueError("model IDs must be unique, nonempty and bounded")
    n = len(models)
    for key, shape in (("W", (n, 1024)), ("b", (n,)), ("mean_in_tokens", (n,)),
                       ("mean_out_tokens", (n,)), ("lam", ())):
        a = z[key]
        if a.shape != shape or a.dtype.kind not in "fi" or not np.isfinite(a).all():
            raise ValueError(f"invalid {key}")
        if key in ("mean_in_tokens", "mean_out_tokens", "lam") and (a < 0).any():
            raise ValueError(f"negative {key}")
    return z


def load(data):
    return validate_head(read_archive(data))


def estimated_cost(z, prices):
    rates = np.asarray([prices[str(m)][:2] for m in z["models"]], dtype=np.float64)
    if not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError("invalid token prices (expected dollars/token)")
    cost = z["mean_in_tokens"] * rates[:, 0] + z["mean_out_tokens"] * rates[:, 1]
    if not np.isfinite(cost).all():
        raise ValueError("nonfinite estimated cost")
    return cost


def cache_key(questions, profile_id=PROFILE_ID):
    return digest({"profile_id": profile_id, "questions": questions})


def save_cache(path, questions, hidden, profile_id=PROFILE_ID):
    np.savez_compressed(path, H=hidden, key=cache_key(questions, profile_id), profile_id=profile_id)


def load_cache(path, questions, profile_id=PROFILE_ID):
    with np.load(path, allow_pickle=False) as z:
        if str(z["profile_id"]) != profile_id or str(z["key"]) != cache_key(questions, profile_id):
            raise ValueError("embedding cache profile/input mismatch")
        h = z["H"]
    if h.shape != (len(questions), 1024) or not np.isfinite(h).all():
        raise ValueError("invalid cached embeddings")
    return h


def validate_manifest(manifest, deployable=False):
    if manifest.get("version") != "fugal-token-statistics-v1":
        raise ValueError("unsupported token manifest")
    if manifest.get("status") not in {"candidate", "reviewed", "synthetic-test-only", "offline-incomplete-provenance"}:
        raise ValueError("unsupported token manifest status")
    if manifest.get("profile_id") != PROFILE_ID:
        raise ValueError("token manifest embedding profile mismatch")
    if not manifest.get("sources") or not manifest.get("worker_profile"):
        raise ValueError("missing observation provenance")
    if deployable and manifest.get("status") != "reviewed":
        raise ValueError("deployable export requires a reviewed token manifest")
    rows = manifest.get("models", [])
    names = [r["id"] for r in rows]
    if not rows or len(names) != len(set(names)):
        raise ValueError("missing or duplicate token observations")
    for row in rows:
        if not isinstance(row["samples"], int) or row["samples"] < 1:
            raise ValueError("missing token observations")
        values = [row["mean_in_tokens"], row["mean_out_tokens"]]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError("invalid token observations")
    return digest(manifest)


def check_manifest(z, manifest, deployable=False):
    identity = validate_manifest(manifest, deployable)
    if scalar(z, "cost_profile_id") != identity:
        raise ValueError("head cost manifest identity mismatch")
    by_id = {r["id"]: r for r in manifest["models"]}
    for key in ("mean_in_tokens", "mean_out_tokens"):
        expected = np.asarray([by_id[str(m)][key] for m in z["models"]])
        if not np.array_equal(z[key], expected):
            raise ValueError("miner token statistics disagree with manifest")


def make_head(W, b, models, manifest, provenance, lam=1.0):
    identity = validate_manifest(manifest)
    rows = {r["id"]: r for r in manifest["models"]}
    z = dict(W=np.asarray(W, dtype=np.float32), b=np.asarray(b, dtype=np.float32),
             models=np.asarray(models), contract=np.asarray(CONTRACT),
             profile_id=np.asarray(PROFILE_ID), backbone_revision=np.asarray(REVISION),
             provenance=np.asarray(provenance), lam=np.asarray(lam, dtype=np.float64),
             cost_profile_id=np.asarray(identity))
    for key in ("mean_in_tokens", "mean_out_tokens"):
        z[key] = np.array([rows[m][key] for m in models], dtype=np.float64)
    return validate_head(z)
