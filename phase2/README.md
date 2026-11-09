# Phase 2 — Legal Named Entity Recognition

**Primary metric:** strict macro-F1  
**Best submission:** A3_V2_NEW + LoRMA → **0.5756** (`p2-a3-v2-new-json.ipynb`, job `lorma`)

## Method

`p2_ner.py` reproduces the Phase-1 normalisation (ASCII-fold + whitespace collapse) while keeping a map back to the original characters, so every BPE token knows its exact `[start, end)`. A gold → BIO → span round trip gives the alignment upper bound (0.989 / 0.985). The causal mask is removed (bidirectional encoder), and a linear head predicts 25 BIO tags on 512-token windows (stride 256) with sqrt class weights. At inference the windows are merged with triangular weights, a BIO-constrained Viterbi pass decodes the tags, and runs are mapped back to whitespace-stripped character spans. Train on fold1, choose the epoch on fold2, refit on fold1+2, predict fold3. Methods: full, LoRMA, LoRA, Houlsby adapter, BitFit, KD (6-layer student from the full model).

## Notes and failures

The first two submissions (0.09 / 0.08) came from an earlier pipeline whose predicted spans did not line up with the original offsets; this led to the offset map and the round-trip check.

## Notebooks

* `A1_V2_NEW`: `p2-a1-v2-new-json.ipynb`
* `A2_V2_PREV`: `p2-a2-v2-old-json.ipynb`
* `A2_V2_NEW`: `p2-a2-new-json-2nd.ipynb`, `p2-a2-new-json-causal.ipynb`
* `A3_V2_OLD`: `p2-a3-old-json.ipynb`, `p2-a3-prev.ipynb`, `p2-a3-v2-old-rerun.ipynb`
* `A3_V2_NEW`: `p2-a3-v2-new-json.ipynb`
* `A4_PREV`: `p2-a4-prev-json.ipynb`

## Codabench scores by backbone and method

| method | A1_V2_NEW | A2_V2_PREV | A2_V2_NEW | A3_V2_OLD | A3_V2_NEW | A4_PREV |
|---|---|---|---|---|---|---|
| full_causal | – | – | 0.4513 | – | – | – |
| kd | 0.509 | – | – | – | 0.5183 | – |
| lora | – | – | 0.5303 | – | – | – |
| lorma | 0.5404 | – | – | – | 0.5756 | 0.5454 |

## How to run

Kaggle, GPU T4 ×2, Internet ON, HF token as secret `HF`. Add the Phase-1 notebook of the same backbone as input. Run once with `QUICK_TEST = True` (smoke test), then *Save & Run All (Commit)* with `QUICK_TEST = False`. Submission zips are written to `/kaggle/working/submissions/` (Phase 1: `/kaggle/working/`). The training code (e.g. `p2_ner.py`, `p4_cjpe.py`) lives in the notebook's `%%writefile` cells and is written to disk when the notebook runs.
