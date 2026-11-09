# Phase 7 — Prior Case Retrieval

**Primary metric:** micro-F1@7 (P@7, R@7 secondary)  
**Best submission:** A2_V2_PREV + KD 2-layer student → **0.4377** (`p7-a2-v2-prev.ipynb`, job `kd_small`)

## Method

`p7_pcr.py`: every query is also in its own pool and is removed from its own ranking. Lexical retrieval (TF-IDF/BM25 grid tuned on dev; best TF-IDF uni+bigram without stop-words) mines hard negatives and is fused with the model. The bi-encoder is built from the Phase-1 encoder: up to 16 chunks of 512 tokens → mean pool → centred, projected to 256-d → unit vector. Training uses InfoNCE (τ=0.05) with 16 queries/step, their positives and 2 BM25 hard negatives per query, with other gold cases masked; GradCache fits it on a T4. Similarity: mean / MaxSim / MaxSim-sym. Final score: `w·z(model) + (1-w)·z(lexical)`, with w tuned on dev and reported with a cross-fitted estimate. Optional temporal filter (decided off on dev). Jobs: lorma, full, kd, kd_small, partial, qlora, ia3, prompt, prefix, frozen (+ ablations of pooling, similarity, objectives, negatives and chunks).

## Notes and failures

TF-IDF alone reaches dev F1@7 0.435, and the best fusion only 0.443. A perfect top-7 can score only about 0.63 because queries have 6–8 relevant cases on average.

## Notebooks

* `A1_V2_NEW`: `p7-a1-v2-new.ipynb`
* `A2_V2_PREV`: `p7-a2-v2-prev.ipynb`
* `A2_V2_NEW`: `p7-a2-v2-newjson.ipynb`
* `A3_V2_OLD`: `p7-a3-v2-old.ipynb`
* `A3_V2_NEW`: `p7-a3-v2-new.ipynb`
* `A4_PREV`: `p7-a4-prev.ipynb`

## Codabench scores by backbone and method

| method | A1_V2_NEW | A2_V2_PREV | A2_V2_NEW | A3_V2_OLD | A3_V2_NEW | A4_PREV |
|---|---|---|---|---|---|---|
| full | 0.433 | 0.4347 | 0.4282 | 0.4294 | 0.433 | 0.4312 |
| ia3 | 0.4318 | 0.4347 | 0.4294 | 0.4312 | 0.4294 | 0.433 |
| kd | 0.43 | 0.4353 | 0.4312 | 0.4306 | 0.4318 | 0.4341 |
| kd_small | 0.4306 | 0.4377 | 0.4324 | 0.4312 | 0.4312 | 0.433 |
| lorma | 0.4335 | 0.4265 | 0.4282 | 0.4341 | 0.4341 | 0.433 |
| partial | 0.4312 | 0.4347 | 0.4335 | 0.4324 | 0.4335 | 0.4335 |
| prompt | 0.4306 | 0.4294 | 0.4282 | 0.43 | 0.433 | 0.4312 |
| qlora | 0.4341 | 0.4347 | 0.433 | 0.433 | 0.4341 | 0.4335 |

## How to run

Kaggle, GPU T4 ×2, Internet ON, HF token as secret `HF`. Add the Phase-1 notebook of the same backbone as input. Run once with `QUICK_TEST = True` (smoke test), then *Save & Run All (Commit)* with `QUICK_TEST = False`. Submission zips are written to `/kaggle/working/submissions/` (Phase 1: `/kaggle/working/`). The training code (e.g. `p2_ner.py`, `p4_cjpe.py`) lives in the notebook's `%%writefile` cells and is written to disk when the notebook runs.
