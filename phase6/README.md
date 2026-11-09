# Phase 6 — Abstractive Summarization

**Primary metric:** BERTScore-F1 (ROUGE-1/L secondary)  
**Best submission:** A3_V2_OLD + LoRA → **0.8465** (`p6-a3-v2-old.ipynb`, job `lora`)

## Method

`p6_summ.py`: 400 training pairs are held out as dev. Extract-then-abstract: a gradient-boosting sentence selector (position, length, TF-IDF centrality, cue words) picks the sentences that best fit a 541-token context. The Phase-1 causal LM (Transformer-XL distances clamped at 511) is trained on `[soft prompt] + context + ' Summary:' + summary` with loss only on the first 480 summary tokens; BERTScore reads only about 400 words. Decoding (n-gram blocking, repetition penalty, top-p) is chosen per job on dev. Jobs: lorma, full, kd (6L), kd_small (4L), lora, qlora, adalora, adapter_h, adapter_p, prompt, prefix, lorma_alt (head truncation ablation), baselines (zero-shot, extractive, lead).

## Notes and failures

The untrained extractive selector scored 0.850 dev BERTScore vs 0.848 for the best fine-tuned model, so the metric rewards copying from the source.

## Notebooks

* `A1_V2_NEW`: `p6-a1-v2-new.ipynb`
* `A2_V2_PREV`: `p6-a2-v2-prev.ipynb`
* `A2_V2_NEW`: `p6-a2-v2-newjson.ipynb`
* `A3_V2_OLD`: `p6-a3-v2-old.ipynb`
* `A3_V2_NEW`: `p6-a3-v2-new.ipynb`
* `A4_PREV`: `p6-a4-prev.ipynb`

## Codabench scores by backbone and method

| method | A1_V2_NEW | A2_V2_PREV | A2_V2_NEW | A3_V2_OLD | A3_V2_NEW | A4_PREV |
|---|---|---|---|---|---|---|
| adalora | 0.8422 | 0.8386 | – | 0.8446 | 0.843 | 0.8366 |
| adapter_h | 0.8416 | – | 0.8192 | – | – | – |
| full | 0.8412 | – | – | 0.8462 | 0.8441 | 0.837 |
| kd | 0.8419 | 0.838 | 0.8158 | 0.8435 | 0.8449 | 0.8337 |
| lora | 0.8422 | – | 0.8175 | 0.8465 | 0.8427 | 0.8359 |
| lorma | 0.8407 | 0.8361 | 0.8168 | 0.8444 | 0.8416 | 0.836 |
| qlora | 0.8422 | 0.8367 | – | 0.845 | 0.8432 | 0.8355 |

## How to run

Kaggle, GPU T4 ×2, Internet ON, HF token as secret `HF`. Add the Phase-1 notebook of the same backbone as input. Run once with `QUICK_TEST = True` (smoke test), then *Save & Run All (Commit)* with `QUICK_TEST = False`. Submission zips are written to `/kaggle/working/submissions/` (Phase 1: `/kaggle/working/`). The training code (e.g. `p2_ner.py`, `p4_cjpe.py`) lives in the notebook's `%%writefile` cells and is written to disk when the notebook runs.
