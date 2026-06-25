"""Cache singleton para o modelo de embeddings SentenceTransformer.

Tanto a Tarefa 02 (chunking) quanto a Tarefa 03 (embeddings) usam o
mesmo modelo all-MiniLM-L6-v2. Este módulo garante que ele seja
carregado na memória apenas uma vez, evitando:

- ~2-3s extras de carregamento na transição entre tarefas
- ~80 MB de RAM duplicada durante a sobreposição

Uso:
    from model_cache import get_sentence_transformer
    model = get_sentence_transformer()
"""

from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_sentence_transformer() -> SentenceTransformer:
    """Retorna o modelo compartilhado de embeddings, carregando-o
    na primeira chamada (lazy initialization)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model
