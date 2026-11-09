# Phase 5 — Legal Statute Identification

**Primary metric:** macro-F1 (micro-F1, subset accuracy secondary)  
**Best submission:** A1_V2_NEW + KD 4-layer student → **0.3907** (`p5-a1-v2-new.ipynb`, job `kd_small`)

## Method

`p5_lsi.py`: the whole document is tokenised and split into 512-token chunks. Up to 16 are kept (first, last, the chunks with the most `<SECTION>`/`<ACT>` masks, evenly spread ones), covering 97% of documents fully. Chunks → Phase-1 encoder + method → BiGRU → label-wise attention (one query per statute) + max pooling → 100 sigmoids. Loss: BCE with `pos_weight = clip(sqrt(neg/pos), 1, 8)`. Global vs per-statute thresholds are compared by 2-fold cross-fitting on dev; every document gets at least 1 statute. Jobs: lorma, full, kd (6L), kd_small (4L), gradual, adalora, qlora, prompt, prefix, adapter, frozen (+ probe ablations of losses, heads and chunk views).

## Notes and failures

Per-statute F1 rises with training frequency (0.32 for <500 docs → 0.58 for ≥3k; r = 0.55).

## Notebooks

* `A1_V2_NEW`: `p5-a1-v2-new.ipynb`
* `A2_V2_PREV`: `p5-a2-v2-prev.ipynb`
* `A2_V2_NEW`: `p5-a2-v2-newjson.ipynb`
* `A3_V2_OLD`: `p5-a3-v2-old.ipynb`
* `A3_V2_NEW`: `p5-a3-v2-new.ipynb`
* `A4_PREV`: `p5-a4-prev.ipynb`

## Codabench scores by backbone and method

| method | A1_V2_NEW | A2_V2_PREV | A2_V2_NEW | A3_V2_OLD | A3_V2_NEW | A4_PREV |
|---|---|---|---|---|---|---|
| kd | 0.3801 | – | – | – | – | – |
| kd_small | 0.3907 | – | – | – | – | – |
| lorma | 0.3719 | – | – | – | – | – |

(P5: per-backbone Codabench submissions were not recorded by backbone except A1; the notebooks print the dev cross-fitted macro-F1 of every job.)

## How to run

Kaggle, GPU T4 ×2, Internet ON, HF token as secret `HF`. Add the Phase-1 notebook of the same backbone as input. Run once with `QUICK_TEST = True` (smoke test), then *Save & Run All (Commit)* with `QUICK_TEST = False`. Submission zips are written to `/kaggle/working/submissions/` (Phase 1: `/kaggle/working/`). The training code (e.g. `p2_ner.py`, `p4_cjpe.py`) lives in the notebook's `%%writefile` cells and is written to disk when the notebook runs.
