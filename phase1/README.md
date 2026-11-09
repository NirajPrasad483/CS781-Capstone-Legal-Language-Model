# Phase 1 — Legal LM Pretraining

**Primary metric:** BERTScore-F1 (ROUGE-L, BLEU-4 secondary)  
**Best submission:** A3_V2_OLD Transformer-XL (64.2M) + MBR → **0.8404** (`a3-v2-oldjson.ipynb`, `submission_mbr.zip`)

## Method

Each notebook streams the four official training files (Acts, High Courts, Supreme Court, Tribunals). It samples documents with fixed per-source weights, caches the tokenised shards, trains one architecture from scratch on 2×T4 (DDP, AdamW β=(0.9,0.95), warmup + cosine, fp16, clip 1.0, checkpoints every 20 min), searches the decoding setting by BERTScore on held-out 512+128-word prompts, and writes `submission.zip` (exactly 128 words per test prompt). The A3 notebooks also run MBR: from 6 sampled candidates plus 1 greedy candidate, keep the one with the highest mean BERTScore against the others. Every notebook asserts 50M ≤ params ≤ 300M.

## Notes and failures

Validation NLL: A1 2.010, A2-PREV 3.109, A2-NEW 2.520 (diverged at 90% of its run, best checkpoint kept), A3-OLD 2.002, A3-NEW 2.041, A4 2.141. Values are per token on each model's own split, so compare them only within a family.

## Notebooks

* `A1_V2_NEW`: `a1-v2-newjson.ipynb`
* `A2_V2_PREV`: `archit-2-0.ipynb`, `archit-2-v2.ipynb`
* `A2_V2_NEW`: `archit-2-new-json.ipynb`
* `A3_V2_OLD`: `a3-v2-oldjson.ipynb`
* `A3_V2_NEW`: `archit-3v2-newjson.ipynb`
* `A4_PREV`: `archit-4-renet.ipynb`

## Codabench scores by backbone and method

| method | A1_V2_NEW | A2_V2_PREV | A2_V2_NEW | A3_V2_OLD | A3_V2_NEW | A4_PREV |
|---|---|---|---|---|---|---|
| MBR | – | – | – | 0.8404 | 0.8339 | – |
| single best decoding | 0.8343 | 0.8351 | 0.8284 | 0.835 | 0.84 | 0.8327 |

## How to run

Kaggle, GPU T4 ×2, Internet ON, HF token as secret `HF`. Attach the tokenizer files as a dataset. Run once with `QUICK_TEST = True` (smoke test), then *Save & Run All (Commit)* with `QUICK_TEST = False`. Submission zips are written to `/kaggle/working/submissions/` (Phase 1: `/kaggle/working/`). The training code (e.g. `p2_ner.py`, `p4_cjpe.py`) lives in the notebook's `%%writefile` cells and is written to disk when the notebook runs.
