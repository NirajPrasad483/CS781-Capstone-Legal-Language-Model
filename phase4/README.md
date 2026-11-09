# Phase 4 — Court Judgment Prediction

**Primary metric:** macro-F1  
**Best submission:** A3_V2_NEW + full fine-tuning → **0.7452** (`p4-a3-v2-new.ipynb`, job `full`)

## Method

`p4_cjpe.py`: the released dev set contains all test cases (with labels) plus duplicates, so every train/dev row that overlaps test is dropped and the set de-duplicated, leaving 31,677 train cases and 994 clean dev cases (497/497). Test labels are never read. Each case keeps the first 512-token chunk + the last 7 chunks; each chunk → Phase-1 encoder (bidirectional) + method → mean pool + chunk-position embedding → BiGRU → attention + max pooling → P(1). Training samples 4 chunks per case (always the last); the threshold is tuned on dev. Jobs: lorma (attention projections, s=1), lorma_all, kd (6L), kd_small (4L), adalora, bitfit, adapter, partial, full, gradual, frozen. Section ablation: last chunks carry the signal.

## Notes and failures

LoRMA on RetNet residual projections (s=2) overflowed fp16 and stuck at chance, so LoRMA was restricted to attention-side projections with s=1. A job still at chance after 10% of training stops itself.

## Notebooks

* `A1_V2_NEW`: `p4-a1-v2-new.ipynb`
* `A2_V2_PREV`: `p4-a2-v2-prev.ipynb`
* `A2_V2_NEW`: `p4-a2-v2-newjson.ipynb`
* `A3_V2_OLD`: `p4-a3-v2-old.ipynb`
* `A3_V2_NEW`: `p4-a3-v2-new.ipynb`
* `A4_PREV`: `p4-a4-prev-2-0.ipynb`, `p4-a4-prev-2-1.ipynb`, `p4-a4-prev1-0.ipynb`, `p4-a4-prev1-1.ipynb`

## Codabench scores by backbone and method

| method | A1_V2_NEW | A2_V2_PREV | A2_V2_NEW | A3_V2_OLD | A3_V2_NEW | A4_PREV |
|---|---|---|---|---|---|---|
| adalora | 0.6784 | – | 0.5807 | 0.6695 | 0.6498 | 0.66 |
| adapter | – | – | – | – | – | 0.7178 |
| bitfit | 0.6402 | – | – | – | 0.653 | 0.6406 |
| frozen | – | – | – | – | 0.5881 | 0.5732 |
| full | 0.6853 | – | 0.6189 | 0.6744 | 0.7452 | 0.7211 |
| gradual | 0.6835 | 0.6577 | 0.7185 | 0.732 | 0.7281 | 0.7002 |
| kd | 0.6954 | 0.6561 | 0.6106 | 0.6995 | 0.696 | 0.7094 |
| kd_small | – | – | – | – | 0.6967 | 0.7112 |
| lorma | 0.7276 | 0.6544 | – | – | 0.7246 | 0.6575 |
| lorma_all | – | – | 0.6214 | 0.6906 | 0.7103 | 0.6801 |
| partial | 0.6932 | 0.6532 | – | 0.7253 | 0.7071 | 0.7321 |

## How to run

Kaggle, GPU T4 ×2, Internet ON, HF token as secret `HF`. Add the Phase-1 notebook of the same backbone as input. Run once with `QUICK_TEST = True` (smoke test), then *Save & Run All (Commit)* with `QUICK_TEST = False`. Submission zips are written to `/kaggle/working/submissions/` (Phase 1: `/kaggle/working/`). The training code (e.g. `p2_ner.py`, `p4_cjpe.py`) lives in the notebook's `%%writefile` cells and is written to disk when the notebook runs.
