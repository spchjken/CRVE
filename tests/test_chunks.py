"""Chunk contract of the context sidecar: about 100 sub-word tokens, boundary within [85, 115] preferring sentence ends,
never more than 115 tokens, chunks contiguous and covering the document; S2 keeps the 20 lowest-IDF-density chunks."""
import re

from crve.pipeline.sidecar import CAP, keep
from crve.selection.gate2_contracts import segment_document_canonical


class WordTokenizer:
    """One token per whitespace-separated word, with character offsets (like a fast tokenizer)."""

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=False, truncation=False):
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out = {"input_ids": list(range(len(spans)))}
        if return_offsets_mapping:
            out["offset_mapping"] = spans
        return out


def _doc(n_words, sentence_every):
    words = []
    for i in range(n_words):
        w = f"w{i}"
        if (i + 1) % sentence_every == 0:
            w += "."
        words.append(w)
    return " ".join(words)


def test_chunks_bounded_contiguous_and_complete():
    text = _doc(1037, 7)
    chunks = segment_document_canonical("d1", text, WordTokenizer())
    assert len(chunks) > 5
    assert all(c.tok_count <= 115 for c in chunks)
    assert all(85 <= c.tok_count <= 115 for c in chunks[:-1])
    for a, b in zip(chunks, chunks[1:]):
        assert text[a.char_end:b.char_start].strip() == ""          # contiguous: only whitespace between chunks
    assert chunks[0].char_start == 0 and chunks[-1].char_end == len(text)
    assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_id == f"d1_c{c.chunk_idx}" for c in chunks)


def test_sentence_end_preferred_nearest_target():
    text = _doc(400, 7)
    first = segment_document_canonical("d", text, WordTokenizer())[0]
    assert first.text.endswith(".")
    # sentence ends fall after words 7, 14, ...: the one nearest to 100 tokens is 98
    assert first.tok_count == 98


def test_short_document_is_one_chunk():
    chunks = segment_document_canonical("d", "a short document.", WordTokenizer())
    assert len(chunks) == 1 and chunks[0].text == "a short document." and chunks[0].tok_count == 3


def test_s2_heap_keeps_lowest_idf_density_then_earliest():
    heap = []
    dens = [5.0, 1.0, 3.0, 1.0] + [9.0] * 30 + [0.5]
    for k, d in enumerate(dens):
        keep(heap, (-d, -k), f"chunk{k}")
    kept = [item for _, item in sorted(heap, key=lambda x: x[0], reverse=True)]
    assert len(kept) == CAP == 20
    assert kept[:5] == ["chunk34", "chunk1", "chunk3", "chunk2", "chunk0"]   # lowest density first; ties: earlier chunk first
    assert all(item in kept for item in [f"chunk{k}" for k in range(4, 19)])  # among equal densities the earliest are kept
    assert "chunk19" not in kept
