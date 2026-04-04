from sentence_transformers import CrossEncoder
import numpy as np

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
    return _reranker

def rerank(query: str, documents: list[str], top_k: int = 3) -> list[str]:
    if not documents or len(documents) <= top_k:
        return documents
    
    try:
        model = get_reranker()
        pairs = [(query, doc[:512]) for doc in documents]
        scores = model.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1]
        return [documents[i] for i in sorted_indices[:top_k]]
    except Exception as e:
        print(f"Reranking failed: {e}")
        return documents[:top_k]