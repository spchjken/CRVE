"""Canonical chunking of documents for the Gate 2 context sidecar.

Every document is cut into contiguous, non-overlapping chunks of about 100 sub-word tokens of the encoder's tokeniser,
independently of any candidate term: each chunk ends at the sentence end, or failing that the word boundary, nearest to
100 tokens within a window of +-15 tokens, and is never longer than 115 tokens (segment_document_canonical).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TARGET_CHUNK_SUBWORDS: int = 100
SNAP_WINDOW: int = 15
MAX_CHUNK_SUBWORDS: int = 115
MAX_ENCODER_TOKENS: int = 128  # 115 + 2 special <= 117 <= 128
DEFAULT_RESERVOIR_CAPACITY: int = 20


@dataclass(frozen=True)
class CanonicalChunk:
    """Document-level canonical chunk, constructed independently of candidate terms."""

    chunk_id: str
    doc_id: str
    char_start: int
    char_end: int
    tok_count: int
    text: str
    chunk_idx: int = 0
    is_overlong_compound: bool = False


@dataclass(frozen=True)
class SourceChunkRef:
    """Retained source chunk reference maintaining exact physical identity and provenance."""

    chunk_id: str
    doc_id: str
    chunk_idx: int
    char_start: int
    char_end: int
    tok_count: int
    text: str
    embedding_idx: int = -1
    is_overlong_compound: bool = False


def is_whole_word_boundary(doc_text: str, offsets: list[tuple[int, int]], tok_idx: int) -> bool:
    """Checks whether the boundary between token tok_idx - 1 and token tok_idx is a whole-word boundary.

    A token boundary is at a whole-word boundary if:
    1. tok_idx == 0 or tok_idx == len(offsets).
    2. The end character of token tok_idx - 1 is followed by whitespace in doc_text.
    3. The start character of token tok_idx is preceded by whitespace in doc_text.
    4. Or there is a non-empty character gap between the two tokens containing whitespace.
    """
    if tok_idx <= 0 or tok_idx >= len(offsets):
        return True

    e_prev = offsets[tok_idx - 1][1]
    s_curr = offsets[tok_idx][0]

    # Character gap between tokens containing whitespace
    if s_curr > e_prev:
        return True
    if e_prev < len(doc_text) and doc_text[e_prev].isspace():
        return True
    if s_curr > 0 and doc_text[s_curr - 1].isspace():
        return True

    return False


def segment_document_canonical(
    doc_id: str,
    doc_text: str,
    tokenizer: Any,
    target_subwords: int = TARGET_CHUNK_SUBWORDS,
    max_subwords: int = MAX_CHUNK_SUBWORDS,
    snap_window: int = SNAP_WINDOW,
) -> list[CanonicalChunk]:
    """Segments a document into canonical chunks independently of candidate terms.

    Contract:
    1. Chunks are document-level canonical; they never depend on candidate terms.
    2. Target length is target_subwords (100). Hard bound is <= max_subwords (115).
       With [CLS] and [SEP], encoder input <= 117 <= 128.
    3. Boundaries search within [target - snap_window, min(target + snap_window, max_subwords)],
       i.e., [85, 115] for target 100 and snap 15:
       - Priority 1: Terminal punctuation ('.', '!', '?', '\\n') at a whole-word boundary
         closest to target_subwords.
       - Priority 2: Any whole-word boundary in the window closest to target_subwords.
       - Priority 3: Fallback backward search to start_tok + 1.
       - Priority 4: Isolated compound word exceeding max_subwords (flagged is_overlong_compound=True).
    4. Words (e.g. 'immunohistochemistry') are never split across chunks unless an isolated
       single word exceeds max_subwords.
    5. Encoded length of each chunk text is strictly verified to be <= max_subwords.
    """
    if not doc_text or not doc_text.strip():
        return []

    # Tokenize entire document with character offsets
    encoding = tokenizer(
        doc_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=False,
    )
    input_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    n_tokens = len(input_ids)

    if n_tokens == 0:
        return []

    chunks: list[CanonicalChunk] = []
    chunk_idx = 0
    start_tok = 0

    while start_tok < n_tokens:
        is_overlong = False
        remaining = n_tokens - start_tok
        if remaining <= max_subwords:
            end_tok = n_tokens
        else:
            target_end = start_tok + target_subwords
            window_start = max(start_tok + 1, target_end - snap_window)
            window_end = min(n_tokens, min(target_end + snap_window, start_tok + max_subwords))

            snapped_end = None

            # Priority 1 & 2: Search within [window_start, window_end]
            punct_cands: list[tuple[int, int, int]] = []
            word_cands: list[tuple[int, int, int]] = []
            for cand in range(window_start, window_end + 1):
                if is_whole_word_boundary(doc_text, offsets, cand):
                    s_c, e_c = offsets[cand - 1]
                    tok_str = doc_text[s_c:e_c]
                    dist = abs(cand - target_end)
                    if any(tok_str.endswith(p) for p in [".", "!", "?", "\n"]):
                        punct_cands.append((dist, -cand, cand))
                    else:
                        word_cands.append((dist, -cand, cand))

            if punct_cands:
                punct_cands.sort()
                snapped_end = punct_cands[0][2]
            elif word_cands:
                word_cands.sort()
                snapped_end = word_cands[0][2]

            # Priority 3: Fallback backward search if no boundary in snap window
            if snapped_end is None:
                for cand in range(window_start - 1, start_tok, -1):
                    if is_whole_word_boundary(doc_text, offsets, cand):
                        snapped_end = cand
                        break

            # Priority 4: Explicit Overlong-Word Policy
            if snapped_end is None:
                snapped_end = min(n_tokens, start_tok + max_subwords)
                is_overlong = True

            end_tok = max(start_tok + 1, snapped_end)

        # Extract text slice
        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]
        chunk_text = doc_text[char_start:char_end]

        # Verify length of the actual chunk text with the tokenizer
        chunk_enc = tokenizer(chunk_text, add_special_tokens=False)
        actual_tok_count = len(chunk_enc["input_ids"])

        # Boundary-preserving shrinking loop:
        while actual_tok_count > max_subwords and end_tok > start_tok + 1:
            prev_boundary = None
            for cand in range(end_tok - 1, start_tok, -1):
                if is_whole_word_boundary(doc_text, offsets, cand):
                    prev_boundary = cand
                    break

            if prev_boundary is not None:
                end_tok = prev_boundary
            else:
                end_tok -= 1

            char_end = offsets[end_tok - 1][1]
            chunk_text = doc_text[char_start:char_end]
            chunk_enc = tokenizer(chunk_text, add_special_tokens=False)
            actual_tok_count = len(chunk_enc["input_ids"])

        chunks.append(
            CanonicalChunk(
                chunk_id=f"{doc_id}_c{chunk_idx}",
                doc_id=doc_id,
                chunk_idx=chunk_idx,
                char_start=char_start,
                char_end=char_end,
                tok_count=actual_tok_count,
                text=chunk_text,
                is_overlong_compound=is_overlong,
            )
        )
        chunk_idx += 1
        start_tok = end_tok

    return chunks
