# Success-head contract v1

`fugal-success-v1` means: question → independent sigmoid success predictions →
subtract λ × estimated dollars per query → round utilities to 1e-4 → select the
first maximum in original artifact row order. Model subsets preserve that order,
even if a caller lists the subset differently. λ changes utility, never the
success predictions. Prices are dollars per token at the Python interface;
JSON price sheets use dollars per million tokens.

The reference implementation is `fugal/success_contract.py`. Subnet vendors this
small NumPy module, its Apache-2.0 license, and its exact source commit/hash.
It has no dependency on the serving application.

## Embeddings

The profile pins Qwen/Qwen3-0.6B model **and tokenizer** to
`c1899de289a04d12100db370d81485cdf75e47ca`. The input is exactly:

```text
system: You are a routing model. Given a question, your hidden state will be used to predict which language model can best answer it. Read the question carefully.
user: {question}
```

Questions are standalone strings (including empty, Unicode, or multiline strings).
The **total** input limit is 2048 tokens, including the routing prompt. Tokenization
uses right truncation, right batch padding, the attention mask for mean pooling,
and then L2 normalization. CPU float32 is the reference execution environment.
The answering model receives the full original request and history through the
existing one-worker path; only the routing representation is truncated.

Both default and custom local model directories are checked against exact SHA256
hashes of model weights, configuration, and tokenizer files before use. A directory
name or revision marker is insufficient. Obtain the pinned snapshot with the
existing fetch script. Conformance is tested at `atol=1e-5, rtol=1e-5` for single
versus batched embeddings, plus exact selection checks on fixed fixtures.

Caches contain both the profile identity and a hash of the ordered questions.
Caches from legacy prompts or the experimental 512-token variant are rejected.

## Artifact and loading

NPZ arrays are loaded with `allow_pickle=False`. Both archive size (1 MiB) and
expanded size (8 MiB), NPY headers, shape claims, numeric finiteness, unique model
IDs, and metadata are checked before routing. Maximum 64 models and 256 characters
per ID. Required fields, with no extra fields accepted:

| Field | Shape / meaning |
| --- | --- |
| W | (models, 1024), finite numeric weights |
| b | (models,), finite numeric biases |
| models | Unique, ordered Unicode IDs |
| contract | Unicode scalar `fugal-success-v1` |
| profile_id | SHA256 of the canonical embedding profile |
| backbone_revision | Pinned HF commit above |
| provenance | Nonempty Unicode scalar describing training and sources |
| lam | Finite nonnegative scalar, default λ |
| mean_in_tokens, mean_out_tokens | Aligned finite nonnegative measured means |
| cost_profile_id | SHA256 of a versioned token observation manifest |

Core checks artifact compatibility and computes estimated cost from these means
and the selected price sheet. Subnet additionally checks the means exactly against
its pinned reviewed benchmark manifest. Core loading alone does not establish the
truth of miner-supplied observation claims.

Core v1/v2 heads retain their existing embedding and selection behavior. The shipped
`data/router_head.npz` is unchanged. Presence of `contract` invokes strict success
validation; an unknown contract cannot fall back to legacy interpretation.

```bash
FUGAL_HEAD=/path/to/bundle/head.npz \
FUGAL_PRICES=/path/to/bundle/prices.json \
FUGAL_MODEL=/path/to/pinned/Qwen3-0.6B \
python -m fugal --route "what is 15% of 240?"
```

Use the bundle price snapshot to reproduce evaluated selections. Explicitly
pointing `FUGAL_PRICES` at refreshed prices is supported, and can change selections.
Core retains `router_lambda`, `FUGAL_LAMBDA`, and model-subset overrides. The live
subnet benchmark always uses λ=1 and its pinned prices.

`verify/verify_head.py` checks success compatibility through the serving loader.
`verify/verify_calibration.py` uses the serving embeddings and shared selector.
`verify/verify_success_contract.py --subnet /path/to/exact/subnet/checkout` checks
real-token and embedding conformance, and accepts `--head` and `--prices` for a
subnet-exported fixture. These commands make no paid worker calls.

**Successful loading proves compatibility, not calibrated probabilities or improved
routing.** The offline SPROUT mechanism report belongs to the companion subnet PR.
Its historical model coverage, request profiles, and price assumptions do not
establish live-subnet calibration. Winner selection and publication stay manual.
