import heapq
import json
import re
from collections import Counter, defaultdict

SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>"]
N_SPECIAL = len(SPECIAL_TOKENS)
N_BYTES = 256
BASE_VOCAB = N_SPECIAL + N_BYTES  
_PRETOKEN_RE = re.compile(r" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+")


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(m) for m in (merges or [])]
        self._build_tables()

    def _build_tables(self):
        self.id_to_bytes = [b""] * N_SPECIAL + [bytes([i]) for i in range(N_BYTES)]
        self.itos = list(SPECIAL_TOKENS) + [self._show(bytes([i])) for i in range(N_BYTES)]
        self.merge_rank = {}
        for rank, (a, b) in enumerate(self.merges):
            self.merge_rank[(a, b)] = rank
            merged = self.id_to_bytes[a] + self.id_to_bytes[b]
            self.id_to_bytes.append(merged)
            self.itos.append(self._show(merged))
        self.vocab_size = len(self.id_to_bytes)
        self._cache = {}

    @staticmethod
    def _show(bs):
        """Human readable form of a byte string (only used for the saved vocab list)."""
        return bs.decode("utf-8", errors="replace")

    def get_vocab_size(self):
        return self.vocab_size

    def _encode_pretoken(self, piece):
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = [N_SPECIAL + b for b in piece.encode("utf-8")]
        rank = self.merge_rank
        while len(ids) > 1:
            best_rank, best_pos = None, -1
            for i in range(len(ids) - 1):
                r = rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pos = r, i
            if best_rank is None:
                break
            a, b = ids[best_pos], ids[best_pos + 1]
            new_id = BASE_VOCAB + best_rank
            out, i = [], 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        self._cache[piece] = ids
        return ids

    def encode(self, text):
        out = []
        for piece in _PRETOKEN_RE.findall(text):
            out.extend(self._encode_pretoken(piece))
        return out

    def decode(self, ids):
        data = b"".join(self.id_to_bytes[int(i)] for i in ids)  
        return data.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"merges": self.merges, "itos": self.itos,
                       "vocab_size": self.vocab_size}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(merges=[tuple(m) for m in d["merges"]])


def _train_merges(text, vocab_size):
    """Learn (vocab_size - 260) merges.  Incremental pair counting + a lazy heap, so
    training on a multi-million-character corpus takes seconds, not hours."""
    n_merges = max(0, vocab_size - BASE_VOCAB)

    freq = Counter(_PRETOKEN_RE.findall(text))
    words = [[N_SPECIAL + b for b in w.encode("utf-8")] for w in freq]
    wfreq = list(freq.values())

    id_to_bytes = [b""] * N_SPECIAL + [bytes([i]) for i in range(N_BYTES)]

    pair_count = defaultdict(int)
    pair_words = defaultdict(set)
    for wi, w in enumerate(words):
        for p in zip(w, w[1:]):
            pair_count[p] += wfreq[wi]
            pair_words[p].add(wi)

    def heap_item(p):
        return (-pair_count[p], id_to_bytes[p[0]], id_to_bytes[p[1]], p)

    heap = [heap_item(p) for p in pair_count]
    heapq.heapify(heap)

    merges = []
    while len(merges) < n_merges and heap:
        negc, _, _, pair = heapq.heappop(heap)
        cur = pair_count.get(pair, 0)
        if cur <= 0 or -negc != cur:
            continue  
        if cur < 2 and len(merges) > 0:
            break
        a, b = pair
        new_id = BASE_VOCAB + len(merges)
        merges.append(pair)
        id_to_bytes.append(id_to_bytes[a] + id_to_bytes[b])

        changed = set()
        for wi in list(pair_words[pair]):
            w, f = words[wi], wfreq[wi]
            for p in zip(w, w[1:]):
                pair_count[p] -= f
                if pair_count[p] <= 0:
                    del pair_count[p]
                changed.add(p)
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            words[wi] = out
            for p in zip(out, out[1:]):
                pair_count[p] += f
                pair_words[p].add(wi)
                changed.add(p)
        pair_words.pop(pair, None)
        pair_count.pop(pair, None)
        changed.discard(pair)
        for p in changed:
            if p in pair_count:
                heapq.heappush(heap, heap_item(p))
    return merges


def train_tokenizer(train_text: str, vocab_size: int):
    """Train the byte-level BPE on `train_text` and return the trained tokenizer."""
    merges = _train_merges(train_text, vocab_size)
    return BPETokenizer(merges)
