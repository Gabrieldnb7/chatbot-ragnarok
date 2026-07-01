# Tarefa 05: Busca semântica com MMR (Maximal Marginal Relevance)
#
# Resolve Issue #6: retrieve_context com top_k=5 retornava chunks
# apenas por similaridade cosseno, resultando em resultados redundantes
# para perguntas amplas. Agora usamos MMR para diversificar os resultados.

import numpy as np
import chromadb

from model_cache import get_sentence_transformer

# Limiar de similaridade (0 a 1, maior = mais relevante).
# Chunks com score abaixo deste valor são marcados como evidência insuficiente.
# Este valor é CONSISTENTE com o DEFAULT_SCORE_THRESHOLD do llm_integration.py.
# Retrieval deve ser ABRANGENTE (trazer candidatos) — o LLM decide o corte final.
REFUSE_THRESHOLD = 0.12

# ── Parâmetros MMR ──────────────────────────────────────────────────
# N candidatos brutos buscados no ChromaDB antes da seleção MMR.
MMR_CANDIDATES = 15

# λ (lambda) controla o trade-off entre relevância e diversidade:
#   λ = 1.0 → 100% relevância (comportamento original, sem diversidade)
#   λ = 0.0 → 100% diversidade (ignora relevância)
#   λ = 0.65 → equilíbrio empírico bom para o domínio PGD
MMR_LAMBDA = 0.65

_chroma_client = None


def _get_collection():
    global _chroma_client
    if _chroma_client is None:
        from chromadb.config import Settings
        from pathlib import Path

        db_path = str(
            Path(__file__).resolve().parent.parent
            / "data"
            / "vector_db"
            / "chroma_data"
        )
        _chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

    try:
        return _chroma_client.get_collection(name="ragnarok_knowledge_base")
    except Exception:
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Similaridade cosseno entre dois vetores normalizados ou não."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _apply_mmr(
    query_embedding: np.ndarray,
    candidate_embeddings: list[np.ndarray],
    candidate_scores: list[float],
    top_k: int,
    lambda_param: float = MMR_LAMBDA,
) -> list[int]:
    """
    Seleciona top_k índices dos candidatos usando MMR.

    MMR = λ · sim(query, candidato) - (1-λ) · max(sim(candidato, já_selecionado))

    Isto garante que os resultados selecionados sejam tanto relevantes
    para a query quanto diversos entre si, evitando redundância temática.

    Parâmetros:
        query_embedding: Vetor da pergunta do usuário.
        candidate_embeddings: Lista de vetores dos chunks candidatos.
        candidate_scores: Similaridade normalizada (0-1) de cada candidato com a query.
        top_k: Número de resultados a selecionar.
        lambda_param: Peso de relevância vs diversidade (0-1).

    Retorno:
        Lista de índices selecionados (na ordem de seleção MMR).
    """
    if not candidate_embeddings:
        return []

    n = len(candidate_embeddings)
    top_k = min(top_k, n)

    selected_indices: list[int] = []
    remaining = set(range(n))

    for _ in range(top_k):
        best_idx = -1
        best_mmr = float("-inf")

        for idx in remaining:
            # Componente de relevância: sim(query, candidato)
            relevance = candidate_scores[idx]

            # Componente de diversidade: max(sim(candidato, já_selecionados))
            if selected_indices:
                max_sim_to_selected = max(
                    _cosine_similarity(
                        candidate_embeddings[idx], candidate_embeddings[sel]
                    )
                    for sel in selected_indices
                )
            else:
                max_sim_to_selected = 0.0

            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return selected_indices


def retrieve_context(
    query: str,
    top_k: int = 5,
    use_mmr: bool = True,
    mmr_lambda: float = MMR_LAMBDA,
    mmr_candidates: int = MMR_CANDIDATES,
) -> list:
    """
    Recupera os chunks mais relevantes e diversos para a pergunta do usuário.

    Quando ``use_mmr=True`` (padrão), busca ``mmr_candidates`` resultados brutos
    do ChromaDB e aplica MMR para selecionar ``top_k`` chunks que maximizam
    tanto a relevância com a query quanto a diversidade entre si.

    O ChromaDB retorna distância cosseno bruta (0 = idêntico, 2 = oposto).
    Esta função converte para SIMILARIDADE (0 a 1, maior = melhor) e
    expõe o resultado como ``score`` — o padrão que o resto do pipeline
    (llm_integration, interface) espera consumir.

    Parâmetros:
        query (str): Pergunta original do usuário.
        top_k (int): Número de chunks desejados (padrão: 5).
        use_mmr (bool): Se True, aplica MMR para diversificação (padrão: True).
        mmr_lambda (float): Peso relevância vs diversidade (padrão: 0.65).
        mmr_candidates (int): Número de candidatos brutos para MMR (padrão: 15).

    Retorno:
        list: Lista de dicionários, cada um contendo:
            - id: Identificador do chunk
            - texto: Conteúdo textual do chunk
            - metadata: Metadados do documento de origem
            - score: Similaridade normalizada (0 a 1, maior = melhor)
            - score_distancia: Distância cosseno bruta do ChromaDB (debug)
            - evidencia_suficiente: bool
            - refuse_motivo: str (se evidencia_suficiente for False)
            - metodo: "mmr" ou "cosine" (indica qual método selecionou o chunk)
    """
    collection = _get_collection()
    if collection is None:
        return []

    # Passo A: Embedding da query usando o singleton compartilhado
    embed_model = get_sentence_transformer()
    query_embedding = embed_model.encode(query)
    query_embedding_list = query_embedding.tolist()

    # Passo B: Busca bruta no banco vetorial
    # Se MMR está ativo, buscamos mais candidatos para ter pool de diversificação
    n_fetch = mmr_candidates if use_mmr else top_k
    # Garantir que não pedimos mais do que a collection tem
    collection_count = collection.count()
    n_fetch = min(n_fetch, collection_count) if collection_count > 0 else top_k

    results = collection.query(
        query_embeddings=[query_embedding_list],
        n_results=n_fetch,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids_list = results.get("ids", [[]])[0]
    embeddings_list = results.get("embeddings", [[]])[0]

    if not documents:
        return []

    # Passo C: Converter distâncias para scores de similaridade
    scores = [round(1.0 - (d / 2.0), 4) for d in distances]

    # Passo D: Selecionar índices — MMR ou top-k simples
    if use_mmr and len(documents) > top_k:
        # Converter embeddings para numpy arrays
        candidate_embs = [np.array(e) for e in embeddings_list]
        query_emb_np = np.array(query_embedding)

        selected_indices = _apply_mmr(
            query_embedding=query_emb_np,
            candidate_embeddings=candidate_embs,
            candidate_scores=scores,
            top_k=top_k,
            lambda_param=mmr_lambda,
        )
        metodo = "mmr"
    else:
        selected_indices = list(range(min(top_k, len(documents))))
        metodo = "cosine"

    # Passo E: Montar resultado final
    recuperados = []
    for i in selected_indices:
        distancia = distances[i]
        score = scores[i]

        chunk_data = {
            "id": ids_list[i],
            "texto": documents[i],
            "metadata": metadatas[i],
            "score": score,
            "score_distancia": round(distancia, 4),
            "metodo": metodo,
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
