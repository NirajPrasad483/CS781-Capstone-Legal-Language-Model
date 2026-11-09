# CS781 Capstone — In-House Legal Intelligence System

**Nirajkumar Binod Prasad · Roll 240701 · IIT Kanpur · Codabench `n_240701`**  
Repository: https://github.com/NirajPrasad483/CS781-Capstone-Legal-Language-Model

This repository holds all code for the CS-781 Legal Language Modelling capstone. Six compact language models were pretrained from scratch on the official Indian legal corpora (Phase 1). Each one was then adapted to six downstream legal NLP tasks (Phases 2–7) with many fine-tuning and parameter-efficient (PEFT) methods. Everything runs on free Kaggle GPUs (2×T4); no pretrained LM weights and no external data are used.

The final report is at `report/CS781-CP-Nirajkumar-Binod-Prasad-240701-report.pdf`.

---

## 1. Results (final leaderboards, 9 Nov 2026)

| Phase | Task | Metric | Best score | Rank | Submitted system |
|---|---|---|---|---|---|
| 1 | Legal LM pretraining (512-word prompt → 128-word continuation) | BERTScore-F1 | **0.8404** | 16 | A3_V2_OLD Transformer-XL + MBR decoding |
| 2 | Legal NER (12 entity types, char spans) | strict macro-F1 | **0.5756** | 29 | A3_V2_NEW + LoRMA |
| 3 | Rhetorical role prediction (13 roles) | macro-F1 | **0.5476** | 28 | A2_V2_NEW + LoRMA |
| 4 | Court judgment prediction (0/1) | macro-F1 | **0.7452** | 53 | A3_V2_NEW + full fine-tuning |
| 5 | Legal statute identification (100 statutes, multi-label) | macro-F1 | **0.3907** | 49 | A1_V2_NEW + KD (4-layer student) |
| 6 | Abstractive summarization | BERTScore-F1 | **0.8465** | 35 | A3_V2_OLD + LoRA |
| 7 | Prior case retrieval (top-7) | micro-F1@7 | **0.4377** | 44 | A2_V2_PREV + KD (2-layer student) + TF-IDF fusion |

`results/codabench_scores.csv` lists every Codabench score by phase, backbone and method.

---

## 2. Repository layout

```
phase1/<BACKBONE>/            Phase-1 pretraining notebook + tokenizer files of that backbone
phaseN/<BACKBONE>/            Phase-N notebook(s) for that backbone (N = 2..7)
phaseN/README.md              task, method, experiments and how to run that phase
results/codabench_scores.csv  all Codabench scores (phase, backbone, method)
report/                       final report (PDF + LaTeX source + figures)
```

Backbone names are used in every phase:

| ID | Architecture | Params | Tokenizer | Phase-1 notebook |
|---|---|---|---|---|
| A1_V2_NEW | decoder-only Transformer: 12 layers, d=768, 12 heads, RoPE, RMSNorm, SwiGLU (d_ff=2048), 1024 context | 109.5M | NEW | `phase1/A1_V2_NEW/a1-v2-newjson.ipynb` |
| A2_V2_PREV | encoder–decoder: 6 encoder + 8 decoder layers, d=768, learned positions, pre-LN | 143.7M | PREV | `phase1/A2_V2_PREV/archit-2-0.ipynb` (tokenizer + data), `archit-2-v2.ipynb` (training) |
| A2_V2_NEW | encoder–decoder: 6+6 layers, d=640, "tail prefix" (decoder also sees the last 64 prompt tokens) | 90.3M | NEW | `phase1/A2_V2_NEW/archit-2-new-json.ipynb` |
| A3_V2_OLD | Transformer-XL: 14 layers, d=512, relative positions, 256-token segments + 256-token memory | 64.2M | OLD (with underscore fix) | `phase1/A3_V2_OLD/a3-v2-oldjson.ipynb` |
| A3_V2_NEW | same Transformer-XL | 64.2M | NEW | `phase1/A3_V2_NEW/archit-3v2-newjson.ipynb` |
| A4_PREV | RetNet: 12 layers, d=768, 6 heads, multi-scale retention | 137.9M | OLD | `phase1/A4_PREV/archit-4-renet.ipynb` |

**Tokenizers** (all byte-level BPE, 32k vocabulary):

* **OLD**: from-scratch BPE trainer (`a1_tokenizer.py` + `bpe_tokenizer.json`) with GPT-style pre-tokenization. A later fix splits runs of `_` correctly.
* **PREV**: HuggingFace `ByteLevelBPETokenizer` trained on a 100 MB sample of the official corpus (built in `archit-2-0.ipynb`).
* **NEW**: a second from-scratch implementation, `a1_tokenization.py` + `vocab.json` / `merges.txt`. It uses heap-based incremental merges and NFKC normalisation, deletes control characters, collapses whitespace and does not lowercase. It is trained on documents sampled across each whole file (Acts 0.15 / HC 0.30 / SC 0.30 / Tribunals 0.25). Measured: 1.27 tokens/word, 0% unknown tokens, 100% round-trip fidelity (`tokenizer_stats.json`).

---

## 3. Method summary per phase

All downstream notebooks:

1. load the Phase-1 checkpoint and **exactly** the Phase-1 tokenizer and normalisation;
2. assert that the token IDs and the encoder forward pass are identical to Phase 1 (max |diff| = 0);
3. run one experiment per adaptation method on two GPU queues;
4. pick the best checkpoint on a local held-out split;
5. write `submission_<method>.zip` (with `prediction.json` at the root) and a `results.json` per experiment.

Adaptation methods implemented across the phases:

* full fine-tuning, partial fine-tuning (top 4 blocks), gradual unfreezing, and a frozen probe;
* LoRA, rsLoRA, AdaLoRA (`peft`), and QLoRA (own NF4 quantisation + LoRA);
* **LoRMA** (multiplicative, `y = (I + s·B·A)·W·x`, B = 0 at init), IA³, BitFit, Houlsby/Pfeiffer adapters, prompt tuning and prefix tuning;
* **knowledge distillation** into 6-layer and 4-layer (or 3/2-layer) students. The loss is `α·CE + (1-α)·T²·KL` with T = 2 and α = 0.5, and the teacher is the better of LoRMA and full fine-tuning.

| Phase | Model head / pipeline | Key details |
|---|---|---|
| 1 | Causal LM pretraining (encoder–decoder: prompt → continuation) | AdamW (0.9, 0.95), warmup + cosine, fp16, clip 1.0, DDP on 2×T4, ~8 h. Decoding is chosen by BERTScore on held-out prompts: n-gram blocking, repetition penalty, top-p, and MBR over 7 candidates for A3. |
| 2 | BIO token classification (25 tags) on 512-token windows (stride 256) | Offset map from normalised text back to the original characters (alignment upper bound 0.989/0.985). Windows merged with triangular weights, BIO-constrained Viterbi decoding. Train on fold1, select on fold2, refit on fold1+2. |
| 3 | Sentence encoder (mean pool) + CL/IT domain embedding + document BiLSTM + 13-way head | Sentences truncated at 128 tokens, 48-sentence windows, sqrt class weights. Ablation: no BiLSTM gives 0.359 vs 0.582. |
| 4 | Hierarchical: first chunk + last 7 chunks of 512 tokens → BiGRU + attention/max pooling → P(1) | Dev/test overlap removed (the released dev set contains the test cases), threshold tuned on 994 clean dev cases, section ablation. |
| 5 | Up to 16 chunks → BiGRU → label-wise attention (one query per statute) → 100 sigmoids | BCE with clipped `pos_weight`; global vs per-statute thresholds compared by 2-fold cross-fitting; at least 1 statute per document. |
| 6 | Extract-then-abstract: gradient-boosting sentence selector fills a 541-token context, then the causal LM writes ~480 summary tokens after `Summary:` | Decoding grid chosen on 400 held-out pairs; extractive and zero-shot baselines; consensus ensemble. |
| 7 | Hybrid retrieval: TF-IDF (uni+bigram, no stop-words) + bi-encoder from the Phase-1 encoder (chunks → 256-d unit vector) | InfoNCE (τ = 0.05) with BM25 hard negatives and GradCache; MaxSim-sym similarity; z-score fusion `w·model + (1-w)·lexical` tuned on dev; query removed from its own pool. |

Each `phaseN/README.md` gives more detail.

---

## 4. How to run (Kaggle)

Every notebook is self-contained and was run on **Kaggle, GPU T4 ×2, Internet ON**.

1. **Data access**: add your Hugging Face token as a Kaggle secret named `HF` (or `HF_TOKEN`). The notebooks stream `Exploration-Lab/CS-781-Capstone` (phase configs `phase_1` … `phase_7`).
2. **Phase 1**: open `phase1/<BACKBONE>/<notebook>.ipynb`.
   * For the backbones that use a pre-built tokenizer, attach its files (`a1_tokenizer.py` + `bpe_tokenizer.json`, or `a1_tokenization.py` + `vocab.json` + `merges.txt`) as a Kaggle dataset.
   * Set `QUICK_TEST = True` and run all (about 10–25 min) for a smoke test of every stage.
   * Set `QUICK_TEST = False` and use *Save & Run All (Commit)* for the real run (about 8–10 h).
   * The output holds the checkpoint, `submission.zip` and, for A3, `submission_mbr.zip`.
3. **Phases 2–7**: open `phaseN/<BACKBONE>/<notebook>.ipynb`.
   * Use *Add Input → Your Work → Notebooks* to add the Phase-1 notebook of the same backbone. Its output provides the checkpoint, model code and tokenizer.
   * Run with `QUICK_TEST = True` first, then commit with `QUICK_TEST = False`.
   * In Phases 4–7 the jobs can be split over two accounts with `GROUP = "A"` / `"B"` (about 1.2–1.9 h per job), or run with `GROUP = "ALL"`.
   * Upload the zips from `Output → submissions/`. The summary cell marks each zip `UPLOAD = yes/NO`; collapsed runs are flagged.
4. **Local evaluation**: every notebook builds its own labelled held-out split and validates the submission schema before writing the zip.

Exact checkpoints behind the submitted predictions:

| Phase | Notebook | Job / zip |
|---|---|---|
| 1 | `phase1/A3_V2_OLD/a3-v2-oldjson.ipynb` | `submission_mbr.zip` |
| 2 | `phase2/A3_V2_NEW/p2-a3-v2-new-json.ipynb` | `submission_lorma.zip` |
| 3 | `phase3/A2_V2_NEW/p3-a2-v2-new-rerun.ipynb` | `submission_lorma.zip` |
| 4 | `phase4/A3_V2_NEW/p4-a3-v2-new.ipynb` | `submission_full.zip` |
| 5 | `phase5/A1_V2_NEW/p5-a1-v2-new.ipynb` | `submission_kd_small.zip` |
| 6 | `phase6/A3_V2_OLD/p6-a3-v2-old.ipynb` | `submission_lora.zip` |
| 7 | `phase7/A2_V2_PREV/p7-a2-v2-prev.ipynb` | `submission_kd_small.zip` |

---

## 5. Main findings

* The 64M **Transformer-XL won 4 of 7 phases**. It processed the most tokens per GPU-hour (segment recurrence), which mattered more than parameter count.
* **Decoding** was the biggest lever in Phase 1: n-gram blocking + repetition penalty added +0.02 BERTScore, and MBR another +0.008 on held-out prompts.
* **Document structure** matters most downstream: a BiLSTM over sentences (P3) added +0.22, and the tail chunks carry the outcome (P4).
* **Distilled small students matched or beat their teachers** (P4: 0.736 vs 0.717; P5: 0.394 vs 0.383).
* **Lexical baselines are strong**: TF-IDF is within 0.01 of the best retriever (P7), and an untrained extractive selector beats every fine-tuned summariser on dev BERTScore (P6).

---

## 6. Rules compliance

* Only the official dataset is used. Phase-1 models are trained from scratch, with 50M–300M parameters asserted in code.
* `roberta-large` is used only to compute the BERTScore metric locally.
* LLM usage is declared in the report: code and documentation were written with the help of Claude (Anthropic).

## References

The report bibliography lists all papers (IL-TUR, Transformer-XL, RetNet, RoPE, RMSNorm, SwiGLU, BPE, LoRA, rsLoRA, AdaLoRA, QLoRA, LoRMA, IA³, BitFit, adapters, prompt/prefix tuning, KD, InfoNCE, GradCache, BM25, ColBERT, MBR, label-wise attention, BERTScore, ROUGE, BLEU).
