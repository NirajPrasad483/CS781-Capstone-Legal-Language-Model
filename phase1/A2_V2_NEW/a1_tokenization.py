#!/usr/bin/env python3
"""
a1_tokenization.py  -  Byte-level BPE tokenizer, written FROM SCRATCH, for Indian legal text.

No HuggingFace `tokenizers`, `sentencepiece` or `tiktoken` is used for training, merging,
encoding or decoding. (`tokenizers` is imported ONLY inside the optional comparison in
`evaluate`, to measure our tokenizer against a reference one.)

Pipeline
--------
  1. build-corpus : stream the 4 huge JSON files with ijson, randomly sample documents spread
                    over the WHOLE file, normalize them, drop short/duplicate docs, and write
                    a ~100 MB training corpus + 2000 held-out evaluation documents.
  2. train        : count pre-token ("word") frequencies, then learn BPE merges on the UNIQUE
                    words with incremental pair-count updates (max-heap + lazy invalidation +
                    pair->words index). Checkpoints every 2000 merges. Saves
                    vocab.json, merges.txt, tokenizer_stats.json.
  3. eval         : tokens/word, unk rate, round-trip rate, speed on the held-out docs
                    (optionally compared with an existing tokenizer).

Usage (Kaggle cell or terminal)
-------------------------------
  pip install ijson requests
  python a1_tokenization.py all --out /kaggle/working/tok                  # stream from HuggingFace
  python a1_tokenization.py all --out tok --data-dir /path/with/train_*.json   # local files
  python a1_tokenization.py all --out tok --ref /path/to/old/tokenizer.json    # + comparison
  python a1_tokenization.py selftest                                       # quick sanity checks (no data)

Using the trained tokenizer later (in any model notebook):
    from a1_tokenization import Tokenizer
    tok = Tokenizer.load("/kaggle/working/tok")
    ids = tok.encode(text)            # normalizes first, no special tokens added
    text = tok.decode(ids)

Token ids: <pad>=0, <s>=1, </s>=2, <unk>=3, then the 256 byte symbols (ids 4..259), then merges.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import os
import pickle
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict

# ======================================================================================
# 0.  CONFIGURATION
# ======================================================================================
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]      # ids 0,1,2,3 (exactly as required)
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
N_SPECIAL = len(SPECIAL_TOKENS)
BYTE_BASE = N_SPECIAL                 # byte value b has token id  BYTE_BASE + b
BASE_VOCAB = N_SPECIAL + 256          # 260: specials + 256 single-byte symbols

VOCAB_SIZE = 32_000
SEED = 42
TOTAL_CORPUS_MB = 100                 # ~100 MB of normalized text for tokenizer training
MIX = {"acts": 0.15, "high_courts": 0.30, "supreme_court": 0.30, "tribunals": 0.25}
EVAL_DOCS = 2000                      # held-out docs, same mix, never used for training
EVAL_MAX_CHARS = 50_000               # long held-out docs are cut to this (keeps eval fast)
MIN_WORDS = 100                       # documents shorter than this are skipped
OVERSAMPLE = 2.0                      # sample ~2x the budget, then trim (robust to unknown doc sizes)
MIN_MERGE_FREQ = 2                    # stop early if the best pair occurs fewer times than this
CKPT_EVERY = 2000                     # save merges every N merges (crash-resume)
MAX_DIGITS_PER_PRETOKEN = None        # None = GPT-2 style (whole numbers). Set 3 to split numbers.

HF_BASE = ("https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone/resolve/main/"
           "Phase_1_Pretraining/Train/")
FILENAMES = {k: f"train_{k}.json" for k in MIX}
# Approximate sizes (from the HF page) used only to pick the sampling probability when the
# server does not report Content-Length.
APPROX_FILE_BYTES = {"acts": 729e6, "high_courts": 10.7e9, "supreme_court": 1.32e9, "tribunals": 6.48e9}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ======================================================================================
# 1.  TEXT NORMALIZATION  (deterministic; reused at encode time)
# ======================================================================================
# Characters we delete: control chars (Cc), invisible format chars such as zero-width space /
# BOM (Cf) and lone surrogates (Cs, which cannot be encoded to UTF-8). Whitespace-like
# controls (\t \n \r ...) are NOT deleted here - they are turned into a single space below.
_DELETE_TABLE = {
    cp: None
    for cp in range(sys.maxunicode + 1)
    if unicodedata.category(chr(cp)) in ("Cc", "Cf", "Cs") and not chr(cp).isspace()
}
_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC -> delete control chars -> collapse every whitespace run to ONE space -> strip.
    Does not lowercase and never removes words. Idempotent: normalize(normalize(x)) == normalize(x)."""
    text = text.translate(_DELETE_TABLE)            # delete first so NFKC sees the final text
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text)                    # newlines/tabs/NBSP/repeated spaces -> " "
    return text.strip()


# ======================================================================================
# 2.  PRE-TOKENIZATION (GPT-2 style) + BYTE <-> UNICODE TABLE
# ======================================================================================
def _build_pretoken_regex(max_digits=None) -> "re.Pattern":
    """GPT-2 pattern, written with the stdlib `re` module (no \\p{L}):
         contractions | optional-space + letters | optional-space + numbers
         | optional-space + other symbols | whitespace | catch-all single char
       The final `.` guarantees that NO character is ever dropped (round-trip safety)."""
    num = r"\d+" if not max_digits else rf"\d{{1,{max_digits}}}"
    pattern = (r"'(?:s|t|re|ve|m|ll|d)"      # contractions: 's 't 're 've 'm 'll 'd
               r"| ?[^\W\d_]+"               # letters (any language)
               rf"| ?{num}"                  # numbers
               r"| ?(?:[^\s\w]|_)+"          # punctuation / symbols
               r"|\s+(?!\S)|\s+"             # whitespace
               r"|.")                        # anything else
    return re.compile(pattern, re.DOTALL)


PRETOKEN_RE = _build_pretoken_regex(MAX_DIGITS_PER_PRETOKEN)


def bytes_to_unicode() -> dict:
    """The GPT-2 table: map each of the 256 byte values to a printable unicode character, so
    every token is a readable string with no spaces/control chars (needed for vocab.json and
    merges.txt). Space (0x20) becomes 'Ġ', newline becomes 'Ċ', etc."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}


# ======================================================================================
# 3.  THE TOKENIZER (encode / decode / save / load)
# ======================================================================================
class Tokenizer:
    """Byte-level BPE tokenizer. Internally merges are stored on integer ids for speed;
    on disk they are stored in the HuggingFace-compatible vocab.json + merges.txt."""

    def __init__(self, id_to_token: list, ranks: dict):
        self.id_to_token = id_to_token                       # id -> token string
        self.token_to_id = {t: i for i, t in enumerate(id_to_token)}
        self.ranks = ranks                                   # (id_a, id_b) -> (rank, merged_id)
        self.vocab_size = len(id_to_token)
        self._cache: dict = {}                               # pre-token string -> list of ids

    # ---------- construction ----------
    @classmethod
    def from_merges(cls, merges: list) -> "Tokenizer":
        """merges: ordered list of (id_a, id_b); merge k creates token id BASE_VOCAB + k."""
        id_to_token = list(SPECIAL_TOKENS) + [BYTE_ENCODER[b] for b in range(256)]
        ranks = {}
        for rank, (a, b) in enumerate(merges):
            id_to_token.append(id_to_token[a] + id_to_token[b])
            ranks[(a, b)] = (rank, BASE_VOCAB + rank)
        return cls(id_to_token, ranks)

    @classmethod
    def load(cls, directory: str) -> "Tokenizer":
        with open(os.path.join(directory, "vocab.json"), encoding="utf-8") as f:
            vocab = json.load(f)
        id_to_token = [None] * len(vocab)
        for tok, i in vocab.items():
            id_to_token[i] = tok
        ranks = {}
        with open(os.path.join(directory, "merges.txt"), encoding="utf-8") as f:
            rank = 0
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#version"):
                    continue
                a, b = line.split(" ")
                ranks[(vocab[a], vocab[b])] = (rank, vocab[a + b])
                rank += 1
        return cls(id_to_token, ranks)

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.token_to_id, f, ensure_ascii=False)
        ordered = sorted(self.ranks.items(), key=lambda kv: kv[1][0])
        with open(os.path.join(directory, "merges.txt"), "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for (a, b), _ in ordered:
                f.write(f"{self.id_to_token[a]} {self.id_to_token[b]}\n")

    # ---------- encoding ----------
    def _encode_word(self, word: str) -> list:
        """BPE on ONE pre-token: start from its bytes, repeatedly apply the lowest-rank
        (= earliest learned) merge that is present, until none applies. Cached per word."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        ids = [BYTE_BASE + b for b in word.encode("utf-8")]
        ranks = self.ranks
        while len(ids) > 1:
            best_rank, best_pos, best_new = None, -1, -1
            for i in range(len(ids) - 1):
                hit = ranks.get((ids[i], ids[i + 1]))
                if hit is not None and (best_rank is None or hit[0] < best_rank):
                    best_rank, best_new, best_pos = hit[0], hit[1], i
            if best_rank is None:
                break
            a, b = ids[best_pos], ids[best_pos + 1]
            out, i = [], 0
            while i < len(ids):                                    # merge ALL occurrences
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(best_new)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        if len(self._cache) > 3_000_000:                           # keep memory bounded
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def encode(self, text: str, normalize: bool = True, add_bos: bool = False,
               add_eos: bool = False) -> list:
        if normalize:
            text = normalize_text(text)
        ids = [BOS_ID] if add_bos else []
        for word in PRETOKEN_RE.findall(text):
            ids.extend(self._encode_word(word))
        if add_eos:
            ids.append(EOS_ID)
        return ids

    # ---------- decoding ----------
    def decode(self, ids, skip_special: bool = True) -> str:
        """ids -> token strings -> concatenate -> map unicode chars back to bytes -> UTF-8.
        Because we concatenate ALL bytes before decoding, multi-byte characters that were split
        across tokens are reassembled correctly."""
        pieces = []
        for i in ids:
            i = int(i)
            if skip_special and i < N_SPECIAL:
                continue
            pieces.append(self.id_to_token[i])
        data = bytes(BYTE_DECODER[c] for c in "".join(pieces))
        return data.decode("utf-8", errors="replace")


# ======================================================================================
# 4.  STREAMING + RANDOM SAMPLING OF THE HUGE JSON FILES
# ======================================================================================
class CountingReader:
    """File-like wrapper that counts bytes read (used for progress / ETA)."""

    def __init__(self, raw):
        self.raw, self.n = raw, 0

    def read(self, size=-1):
        try:
            data = self.raw.read(size)
        except ValueError:
            # urllib3 closes the HTTP response as soon as the last byte arrives, and ijson then
            # asks for one more chunk to detect end-of-file. A closed stream here simply means
            # EOF, so return b"". (A TRUNCATED download is still caught: ijson then raises an
            # incomplete-JSON error and build_corpus retries the source.)
            if getattr(self.raw, "closed", False):
                return b""
            raise
        self.n += len(data)
        return data


def _open_source(src: str):
    """Return (buffered binary stream, total_size_or_None, closer) for a local path or http(s) URL."""
    if src.startswith("http"):
        import requests
        headers = {}
        if os.environ.get("HF_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
        resp = requests.get(src, headers=headers, stream=True, timeout=(30, 300))
        resp.raise_for_status()
        resp.raw.decode_content = True
        size = int(resp.headers["Content-Length"]) if resp.headers.get("Content-Length") else None
        return io.BufferedReader(resp.raw, buffer_size=1 << 22), size, resp.close
    f = open(src, "rb")
    return io.BufferedReader(f, buffer_size=1 << 22), os.path.getsize(src), f.close


def iter_documents(stream, reader):
    """Yield documents one at a time from a JSON list [...] or JSON dict {id: doc} using ijson,
    so even a 10 GB file is never fully in memory. `reader` is a CountingReader around `stream`
    (we peek at the first byte on `stream`, then parse through `reader` to count bytes)."""
    import ijson
    first = stream.peek(1024).lstrip()[:1]
    if first == b"[":
        try:
            return ijson.items(reader, "item", use_float=True)
        except TypeError:                                    # old ijson without use_float
            return ijson.items(reader, "item")
    if first == b"{":
        return (v for _, v in ijson.kvitems(reader, ""))
    raise ValueError("JSON file must start with '[' or '{'")


def _doc_hash(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()


def sample_source(name: str, src: str, train_bytes: int, n_eval: int, seen: set, seed: int):
    """Stream ONE source file once and return (train_docs, eval_docs).

    Random spread over the whole file: every document is kept independently with probability
    p = OVERSAMPLE * budget / file_size  (a Bernoulli sample, so selected docs are uniformly
    spread, not the first N MB). We then shuffle, give the first `n_eval` docs to the held-out
    set and fill the training budget from the rest, so train and eval never overlap."""
    stream, size, closer = _open_source(src)
    size = size or APPROX_FILE_BYTES[name]
    p = min(1.0, OVERSAMPLE * (train_bytes + n_eval * 20_000) / size)
    log(f"[{name}] size={size/1e9:.2f} GB  sampling prob p={p:.5f}  train budget={train_bytes/1e6:.1f} MB")
    rng = random.Random(f"{seed}-{name}")
    reader = CountingReader(stream)
    pool, local_seen = [], set()
    scanned = skipped_short = skipped_dup = 0
    t0 = time.time()
    try:
        for rec in iter_documents(stream, reader):
            scanned += 1
            if scanned % 50_000 == 0:
                done = min(reader.n / size, 0.999)
                el = time.time() - t0
                log(f"[{name}] scanned {scanned:,} docs | {reader.n/1e9:.2f} GB ({done:.0%}) | "
                    f"kept {len(pool):,} | ETA {el/max(done,1e-9)-el:,.0f}s")
            if rng.random() >= p:                            # cheap rejection BEFORE normalizing
                continue
            text = rec if isinstance(rec, str) else (rec.get("text") if isinstance(rec, dict) else None)
            if not isinstance(text, str):
                continue
            text = normalize_text(text)
            if text.count(" ") + 1 < MIN_WORDS:
                skipped_short += 1
                continue
            h = _doc_hash(text)
            if h in seen or h in local_seen:
                skipped_dup += 1
                continue
            local_seen.add(h)
            pool.append(text)
    finally:
        closer()
    seen |= local_seen
    rng.shuffle(pool)
    eval_docs = []
    for t in pool[:n_eval]:
        if len(t) > EVAL_MAX_CHARS:
            t = t[:EVAL_MAX_CHARS].rsplit(" ", 1)[0]
        eval_docs.append(t)
    train_docs, used = [], 0
    for t in pool[n_eval:]:
        if used >= train_bytes:
            break
        train_docs.append(t)
        used += len(t.encode("utf-8"))
    log(f"[{name}] done in {time.time()-t0:,.0f}s: scanned={scanned:,} pool={len(pool):,} "
        f"short={skipped_short:,} dup={skipped_dup:,} -> train {len(train_docs):,} docs/{used/1e6:.1f} MB, "
        f"eval {len(eval_docs):,} docs")
    if used < 0.9 * train_bytes:
        log(f"[{name}] WARNING: got only {used/1e6:.1f} MB of {train_bytes/1e6:.1f} MB - raise OVERSAMPLE")
    return train_docs, eval_docs


def build_corpus(out_dir: str, data_dir: str | None = None, total_mb: float = TOTAL_CORPUS_MB,
                 eval_docs: int = EVAL_DOCS, seed: int = SEED, force: bool = False) -> dict:
    """Create  <out>/corpus/{source}.train.txt (one doc per line)  and  eval_docs.jsonl.
    Each source is cached separately, so a crash only repeats the unfinished source."""
    cdir = os.path.join(out_dir, "corpus")
    os.makedirs(cdir, exist_ok=True)
    seen: set = set()
    all_eval, stats = [], {}
    for name, ratio in MIX.items():
        tr_path = os.path.join(cdir, f"{name}.train.txt")
        ev_path = os.path.join(cdir, f"{name}.eval.jsonl")
        if os.path.exists(tr_path) and os.path.exists(ev_path) and not force:
            log(f"[{name}] cached corpus found - skipping stream")
            with open(tr_path, encoding="utf-8") as f:
                train_docs = [line.rstrip("\n") for line in f]
            with open(ev_path, encoding="utf-8") as f:
                ev = [json.loads(line)["text"] for line in f]
            for t in train_docs + ev:
                seen.add(_doc_hash(t))
        else:
            src = (os.path.join(data_dir, FILENAMES[name]) if data_dir else HF_BASE + FILENAMES[name])
            for attempt in range(1, 4):                      # network hiccups on 10 GB streams
                try:
                    train_docs, ev = sample_source(name, src, int(total_mb * 1e6 * ratio),
                                                   round(eval_docs * ratio), seen, seed)
                    break
                except Exception as e:                       # noqa: BLE001
                    log(f"[{name}] attempt {attempt} failed: {e!r}")
                    if attempt == 3:
                        raise
            with open(tr_path, "w", encoding="utf-8") as f:
                for t in train_docs:
                    f.write(t + "\n")                        # safe: normalized docs contain no newline
            with open(ev_path, "w", encoding="utf-8") as f:
                for t in ev:
                    f.write(json.dumps({"source": name, "text": t}, ensure_ascii=False) + "\n")
        stats[name] = {"train_docs": len(train_docs),
                       "train_mb": round(sum(len(t.encode("utf-8")) for t in train_docs) / 1e6, 2),
                       "eval_docs": len(ev)}
    log(f"corpus stats: {json.dumps(stats)}")
    return stats


# ======================================================================================
# 5.  BPE TRAINING  (incremental pair counts + lazy max-heap + pair->words index)
# ======================================================================================
def count_words(out_dir: str) -> Counter:
    """Pre-tokenize the training corpus and count how often each distinct pre-token ('word')
    occurs. BPE then runs on the (few hundred thousand) UNIQUE words with their counts,
    which is far cheaper than running over ~100 MB of raw text."""
    cache = os.path.join(out_dir, "word_freq.pkl")
    if os.path.exists(cache):
        log("loading cached word frequencies")
        with open(cache, "rb") as f:
            return pickle.load(f)
    freq: Counter = Counter()
    t0, n_docs = time.time(), 0
    cdir = os.path.join(out_dir, "corpus")
    for name in MIX:
        with open(os.path.join(cdir, f"{name}.train.txt"), encoding="utf-8") as f:
            for line in f:
                freq.update(PRETOKEN_RE.findall(line.rstrip("\n")))
                n_docs += 1
    log(f"counted {sum(freq.values()):,} pre-tokens ({len(freq):,} unique) in {n_docs:,} docs, "
        f"{time.time()-t0:.0f}s")
    with open(cache, "wb") as f:
        pickle.dump(freq, f)
    return freq


def learn_bpe(word_freq: Counter, vocab_size: int, out_dir: str) -> list:
    """Learn merges. State:
         words[i]       the current symbol sequence (list of ids) of unique word i
         freqs[i]       how many times word i occurs in the corpus
         pair_count     pair -> total occurrences over the corpus (sum of freqs)
         pair_words     pair -> set of word indices that (may) contain it
         heap           max-heap of (-count, pair); entries can be stale -> checked on pop
       Each step: pop the best pair, rewrite ONLY the words containing it, and update the
       counts of just the pairs that changed (never recount everything)."""
    n_merges = vocab_size - BASE_VOCAB
    words, freqs = [], []
    for w, f in word_freq.items():
        words.append([BYTE_BASE + b for b in w.encode("utf-8")])
        freqs.append(f)

    pair_count: dict = defaultdict(int)
    pair_words: dict = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in zip(w, w[1:]):
            pair_count[p] += f
            pair_words[p].add(wi)
    heap = [(-c, p) for p, c in pair_count.items()]
    heapq.heapify(heap)                                       # ties broken by smaller ids -> deterministic

    ckpt = os.path.join(out_dir, "bpe_ckpt.json")
    forced = []                                               # merges to REPLAY when resuming
    if os.path.exists(ckpt):
        with open(ckpt) as f:
            forced = [tuple(m) for m in json.load(f)["merges"]]
        log(f"resuming: replaying {len(forced):,} merges from checkpoint")

    merges: list = []
    t0 = time.time()
    while len(merges) < n_merges:
        step = len(merges)
        if step < len(forced):
            pair = forced[step]
        else:
            while True:                                       # pop until the entry is up to date
                if not heap:
                    pair = None
                    break
                negc, pair = heapq.heappop(heap)
                if pair_count.get(pair, 0) == -negc:
                    break
            if pair is None or -negc < MIN_MERGE_FREQ:
                log(f"stopping early at {step:,} merges (no pair with freq >= {MIN_MERGE_FREQ})")
                break
        a, b = pair
        new_id = BASE_VOCAB + step
        merges.append(pair)
        best_freq = pair_count.get(pair, 0)                   # frequency of the pair we merge now

        changed = set()
        for wi in pair_words.pop(pair, ()):
            w = words[wi]
            out, i, n, merged = [], 0, len(w), False
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new_id)
                    i += 2
                    merged = True
                else:
                    out.append(w[i])
                    i += 1
            if not merged:                                    # stale index entry
                continue
            f = freqs[wi]
            for p in zip(w, w[1:]):                           # remove old pairs
                pair_count[p] -= f
                changed.add(p)
            for p in zip(out, out[1:]):                       # add new pairs
                pair_count[p] += f
                pair_words[p].add(wi)
                changed.add(p)
            words[wi] = out
        pair_count.pop(pair, None)
        changed.discard(pair)
        for p in changed:
            c = pair_count.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p))                 # fresh entry; older ones become stale
            else:
                pair_count.pop(p, None)

        k = len(merges)
        if k % 500 == 0:
            el = time.time() - t0
            log(f"merge {k:,}/{n_merges:,} | merged-pair freq {best_freq:,} | "
                f"{el:,.0f}s elapsed | ETA {el/k*(n_merges-k):,.0f}s")
        if k % CKPT_EVERY == 0 and k > len(forced):
            tmp = ckpt + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"merges": merges}, f)
            os.replace(tmp, ckpt)                             # atomic: never a half-written ckpt
    log(f"BPE finished: {len(merges):,} merges in {time.time()-t0:,.0f}s")
    return merges


def train(out_dir: str, vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    freq = count_words(out_dir)
    t_count = time.time() - t0
    merges = learn_bpe(freq, vocab_size, out_dir)
    tok = Tokenizer.from_merges(merges)
    tok.save(out_dir)
    stats = {"vocab_size": tok.vocab_size, "n_merges": len(merges), "special_tokens": SPECIAL_TOKENS,
             "pretokenizer_regex": PRETOKEN_RE.pattern, "max_digits_per_pretoken": MAX_DIGITS_PER_PRETOKEN,
             "normalization": "NFKC, delete Cc/Cf/Cs, collapse whitespace to one space, strip; no lowercasing",
             "seed": SEED, "mix": MIX, "unique_words": len(freq), "total_pretokens": sum(freq.values()),
             "word_count_seconds": round(t_count, 1), "total_train_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_dir, "tokenizer_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log(f"saved vocab.json, merges.txt, tokenizer_stats.json to {out_dir} (vocab={tok.vocab_size})")
    return tok


# ======================================================================================
# 6.  EVALUATION ON HELD-OUT DOCUMENTS
# ======================================================================================
def _load_reference(path: str):
    """Optional comparison tokenizer. Accepts a HuggingFace tokenizer.json, a dir with
    vocab.json+merges.txt, or the old a1_tokenizer-style json. Returns fn(text)->ids or None."""
    try:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "tokenizer.json")):
            path = os.path.join(path, "tokenizer.json")
        if os.path.isdir(path):
            from tokenizers import ByteLevelBPETokenizer          # reference only, NOT used for training
            t = ByteLevelBPETokenizer(os.path.join(path, "vocab.json"), os.path.join(path, "merges.txt"))
            return lambda s: t.encode(s).ids
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if "itos" in d:                                          # old hand-written tokenizer
            from a1_tokenizer import BPETokenizer
            t = BPETokenizer.load(path)
            return t.encode
        from tokenizers import Tokenizer as HFTok                # reference only
        t = HFTok.from_file(path)
        return lambda s: t.encode(s, add_special_tokens=False).ids
    except Exception as e:                                       # noqa: BLE001
        log(f"could not load reference tokenizer ({e!r}) - skipping comparison")
        return None


def evaluate(out_dir: str, ref_path: str | None = None) -> dict:
    tok = Tokenizer.load(out_dir)
    docs = []
    for name in MIX:
        with open(os.path.join(out_dir, "corpus", f"{name}.eval.jsonl"), encoding="utf-8") as f:
            docs += [json.loads(line) for line in f]
    docs = docs[:EVAL_DOCS]
    log(f"evaluating on {len(docs):,} held-out docs")

    tok._cache.clear()                                           # cold cache = honest speed
    t0 = time.time()
    encoded = [tok.encode(d["text"], normalize=False) for d in docs]
    enc_time = time.time() - t0

    per_src: dict = defaultdict(lambda: [0, 0, 0])               # source -> [tokens, words, chars]
    n_tok = n_unk = n_pass = n_words = n_chars = 0
    for d, ids in zip(docs, encoded):
        text = d["text"]
        w = text.count(" ") + 1
        n_tok += len(ids)
        n_unk += sum(1 for i in ids if i == UNK_ID)
        n_words += w
        n_chars += len(text)
        n_pass += int(tok.decode(ids) == normalize_text(text))   # exact round trip
        s = per_src[d["source"]]
        s[0] += len(ids); s[1] += w; s[2] += len(text)
    res = {"docs": len(docs), "tokens": n_tok, "words": n_words,
           "tokens_per_word": round(n_tok / n_words, 4), "chars_per_token": round(n_chars / n_tok, 4),
           "unk_rate": n_unk / n_tok, "round_trip_pass_rate": n_pass / len(docs),
           "encode_tokens_per_sec": round(n_tok / enc_time), "encode_MB_per_sec": round(n_chars / 1e6 / enc_time, 3),
           "per_source_tokens_per_word": {k: round(v[0] / v[1], 4) for k, v in per_src.items()}}
    if ref_path:
        ref = _load_reference(ref_path)
        if ref:
            t0 = time.time()
            r_tok = sum(len(ref(d["text"])) for d in docs)
            rt = time.time() - t0
            res["reference"] = {"tokens_per_word": round(r_tok / n_words, 4),
                                "tokens_per_sec": round(r_tok / rt),
                                "relative_token_count_ours_vs_ref": round(n_tok / r_tok, 4)}
    for k, v in res.items():
        log(f"  {k}: {v}")
    sp = os.path.join(out_dir, "tokenizer_stats.json")
    stats = json.load(open(sp)) if os.path.exists(sp) else {}
    stats["eval"] = res
    json.dump(stats, open(sp, "w"), indent=2)
    return res


# ======================================================================================
# 7.  SELF-TEST (no data / network needed)
# ======================================================================================
def selftest() -> None:
    import tempfile
    log("1) normalize_text is idempotent for every unicode code point")
    bad = [cp for cp in range(sys.maxunicode + 1)
           if normalize_text(chr(cp)) != normalize_text(normalize_text(chr(cp)))]
    assert not bad, f"not idempotent for {bad[:5]}"
    assert normalize_text("a\n\n  b\t\u00a0c\u200b d\x00") == "a b c d"
    log("2) train a tiny BPE, check exact round trip incl. unicode / emoji / Devanagari")
    rng = random.Random(0)                                   # varied toy corpus (many distinct words)
    vocab_words = ["".join(rng.choice("abcdefghilmnoprstu") for _ in range(rng.randint(2, 9))) for _ in range(800)]
    corpus = " ".join(rng.choice(vocab_words) for _ in range(30_000)) + " Hon'ble Court Section 14 of 2019."
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "corpus"))
        for n in MIX:
            open(os.path.join(d, "corpus", f"{n}.train.txt"), "w").write(corpus + "\n")
            open(os.path.join(d, "corpus", f"{n}.eval.jsonl"), "w").write(
                json.dumps({"source": n, "text": normalize_text(corpus[:3000])}) + "\n")
        tok = train(d, vocab_size=400)
        assert tok.vocab_size == 400
        tok2 = Tokenizer.load(d)
        tests = ["Hello world", "Hon'ble Supreme Court of India's order dated 26.7.2017",
                 "धारा 302 भारतीय दंड संहिता", "emoji 😀 test ✓", "x" * 50, "a_b__c ((( ))) ...", ""]
        for t in tests:
            ids = tok.encode(t)
            assert tok.decode(ids) == normalize_text(t), t
            assert ids == tok2.encode(t), "save/load mismatch"
            assert UNK_ID not in ids
        assert tok.encode("x", add_bos=True, add_eos=True)[0] == BOS_ID
        res = evaluate(d)
        assert res["round_trip_pass_rate"] == 1.0 and res["unk_rate"] == 0
        first = open(os.path.join(d, "merges.txt")).readline().strip()
        assert first == "#version: 0.2"
    log("ALL SELF-TESTS PASSED")


# ======================================================================================
# 8.  COMMAND LINE
# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build-corpus", "train", "eval", "all", "selftest"])
    ap.add_argument("--out", default="tokenizer_out")
    ap.add_argument("--data-dir", default=None, help="folder with train_*.json (default: stream from HuggingFace)")
    ap.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    ap.add_argument("--total-mb", type=float, default=TOTAL_CORPUS_MB)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ref", default=None, help="existing tokenizer to compare against")
    ap.add_argument("--force", action="store_true", help="rebuild corpus even if cached")
    a = ap.parse_args()
    if a.command == "selftest":
        return selftest()
    if a.command in ("build-corpus", "all"):
        build_corpus(a.out, a.data_dir, a.total_mb, EVAL_DOCS, a.seed, a.force)
    if a.command in ("train", "all"):
        train(a.out, a.vocab_size)
    if a.command in ("eval", "all"):
        evaluate(a.out, a.ref)


if __name__ == "__main__":
    main()
