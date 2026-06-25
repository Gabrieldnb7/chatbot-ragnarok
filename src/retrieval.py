# Tarefa 05: Busca semântica e lógica de recusa

import os
from sentence_transformers import SentenceTransformer
import chromadb

# Limiar de similaridade (0 a 1, maior = mais relevante).
# Chunks com score abaixo deste valor são marcados como evidência insuficiente.
# Este valor é CONSISTENTE com o DEFAULT_SCORE_THRESHOLD do llm_integration.py.
# Retrieval deve ser ABRANGENTE (trazer candidatos) — o LLM decide o corte final.
REFUSE_THRESHOLD = 0.12

_embed_model = None
_chroma_client = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _get_collection():
    global _chroma_client
    if _chroma_client is None:
        from chromadb.config import Settings
        db_path = "../data/vector_db/chroma_data"
        _chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

    try:
        return _chroma_client.get_collection(name="ragnarok_knowledge_base")
    except Exception:
        return None


def retrieve_context(query: str, top_k: int = 3) -> list:
    """
    Recupera os chunks mais relevantes para a pergunta do usuário.

    O ChromaDB retorna distância cosseno bruta (0 = idêntico, 2 = oposto).
    Esta função converte para SIMILARIDADE (0 a 1, maior = melhor) e
    expõe o resultado como ``score`` — o padrão que o resto do pipeline
    (llm_integration, interface) espera consumir.

    Parâmetros:
        query (str): Pergunta original do usuário.
        top_k (int): Número de chunks desejados (padrão: 3).

    Retorno:
        list: Lista de dicionários, cada um contendo:
            - id: Identificador do chunk
            - texto: Conteúdo textual do chunk
            - metadata: Metadados do documento de origem
            - score: Similaridade normalizada (0 a 1, maior = melhor)
            - score_distancia: Distância cosseno bruta do ChromaDB (debug)
            - evidencia_suficiente: bool
            - refuse_motivo: str (se evidencia_suficiente for False)
    """
    collection = _get_collection()
    if collection is None:
        return []

    # Passo A: Embedding da query
    embed_model = _get_embed_model()
    query_embedding = embed_model.encode(query).tolist()

    # Passo B: Busca no banco vetorial
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids_list = results.get("ids", [[]])[0]

    recuperados = []

    for i in range(len(documents)):
        distancia = distances[i]

        # Converte distância cosseno (0 a 2) para similaridade (0 a 1)
        # Fórmula: score = 1 - (distancia / 2)
        #   distância 0.0 → score 1.0 (idêntico)
        #   distância 1.0 → score 0.5 (ortogonal)
        #   distância 2.0 → score 0.0 (oposto)
        score = round(1.0 - (distancia / 2.0), 4)

        chunk_data = {
            "id": ids_list[i],
            "texto": documents[i],
            "metadata": metadatas[i],
            "score": score,
            "score_distancia": round(distancia, 4),
        }

        # Lógica de Refuse: marca chunks com similaridade muito baixa
        if score < REFUSE_THRESHOLD:
            chunk_data["evidencia_suficiente"] = False
            chunk_data["refuse_motivo"] = (
                f"Similaridade {score:.2f} abaixo do limiar {REFUSE_THRESHOLD}"
            )
        else:
            chunk_data["evidencia_suficiente"] = True

        recuperados.append(chunk_data)

    return recuperados
