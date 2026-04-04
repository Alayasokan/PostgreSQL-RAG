from rank_bm25 import BM25Okapi
from db import db
from typing import List, Tuple, Optional

_bm25: Optional[BM25Okapi] = None
_all_chunks: List[str] = []
_last_chunk_count: int = 0

def rebuild_bm25_index() -> None:
    global _bm25, _all_chunks, _last_chunk_count
    cursor = db.execute_sql("SELECT chunk FROM document_information_chunks")
    rows = cursor.fetchall()
    if not rows:
        _bm25 = None
        _all_chunks = []
        _last_chunk_count = 0
        print("BM25 index: no chunks to index")
        return
    _all_chunks = [r[0][:1000] for r in rows]  # limit length for speed
    tokenized = [chunk.split() for chunk in _all_chunks]
    _bm25 = BM25Okapi(tokenized)
    _last_chunk_count = len(rows)
    print(f"BM25 index rebuilt with {_last_chunk_count} chunks")

def get_bm25() -> Tuple[Optional[BM25Okapi], List[str], List[int]]:
    global _bm25, _last_chunk_count
    current_count = db.execute_sql("SELECT COUNT(*) FROM document_information_chunks").fetchone()[0]
    if current_count != _last_chunk_count:
        print(f"Chunk count changed ({_last_chunk_count} -> {current_count}), rebuilding BM25")
        rebuild_bm25_index()
    elif _bm25 is None:
        rebuild_bm25_index()
    # Return chunk IDs for compatibility with existing code (even if unused)
    chunk_ids = list(range(len(_all_chunks)))  # dummy IDs
    return _bm25, _all_chunks, chunk_ids