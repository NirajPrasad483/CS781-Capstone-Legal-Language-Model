# Phase 3 — Rhetorical Role Prediction

**Primary metric:** macro-F1  
**Best submission:** A2_V2_NEW + LoRMA → **0.5476** (`p3-a2-v2-new-rerun.ipynb`, job `lorma`)

## Method

`p3_rr.py`: each sentence (exact Phase-1 tokenisation, ≤128 tokens) → Phase-1 encoder + PEFT → mean-pooled vector + 16-d CL/IT domain embedding → document BiLSTM (256 hidden) → 13-way linear head (sqrt class weights). Documents are processed in 48-sentence windows. Training uses CL+IT train jointly; the best epoch is chosen on dev, then the model is refit on train+dev. Jobs: lorma (also the KD teacher), kd (3-layer student), lora, rslora, ia3, adalora, lorma_noctx (no BiLSTM, ablation), frozen (probe).

## Notes and failures

In the first run, LoRMA/KD with adapter LR 1e-3 collapsed to predicting `Fact` everywhere (dev 0.031). The rerun at 3e-4 reached dev 0.582. Removing document context gives 0.359.

## Notebooks

* `A1_V2_NEW`: `p3-a1-v2-new-rerun.ipynb`, `p3-a1-v2-new.ipynb`
* `A2_V2_PREV`: `p3-a2-v2-prev-rerun.ipynb`, `p3-a2-v2-prev.ipynb`
* `A2_V2_NEW`: `p3-a2-v2-new-final-rerun.ipynb`, `p3-a2-v2-new-rerun.ipynb`, `p3-a2-v2-new.ipynb`
* `A3_V2_OLD`: `p3-a3-v2-old.ipynb`
* `A3_V2_NEW`: `p3-a3-v2-new-rerun.ipynb`, `p3-a3-v2-new.ipynb`
* `A4_PREV`: `p3-a4-prev-actual-rerun.ipynb`, `p3-a4-prev-actual.ipynb`

## Codabench scores by backbone and method

| method | A1_V2_NEW | A2_V2_PREV | A2_V2_NEW | A3_V2_OLD | A3_V2_NEW | A4_PREV |
|---|---|---|---|---|---|---|
| adalora | 0.5223 | 0.4658 | 0.4994 | 0.4762 | 0.4962 | 0.4863 |
| frozen | 0.4185 | 0.4513 | – | 0.4604 | 0.4237 | 0.4467 |
| ia3 | 0.494 | 0.4008 | 0.4497 | 0.518 | 0.4821 | 0.4545 |
| kd | 0.4851 | 0.4434 | 0.4828 | 0.5011 | 0.5303 | 0.4584 |
| lora | 0.435 | 0.4762 | 0.511 | 0.5016 | 0.5193 | 0.435 |
| lorma | 0.491 | 0.4646 | 0.5476 | 0.5154 | 0.5303 | 0.5033 |
| rslora | 0.5091 | 0.4885 | 0.5425 | 0.504 | 0.5333 | 0.4527 |

## How to run

Kaggle, GPU T4 ×2, Internet ON, HF token as secret `HF`. Add the Phase-1 notebook of the same backbone as input. Run once with `QUICK_TEST = True` (smoke test), then *Save & Run All (Commit)* with `QUICK_TEST = False`. Submission zips are written to `/kaggle/working/submissions/` (Phase 1: `/kaggle/working/`). The training code (e.g. `p2_ner.py`, `p4_cjpe.py`) lives in the notebook's `%%writefile` cells and is written to disk when the notebook runs.
