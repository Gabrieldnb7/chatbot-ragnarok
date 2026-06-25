# Tarefa 05: Busca semântica e lógica de recusa

import os
from sentence_transformers import SentenceTransformer
import chromadb

REFUSE_THRESHOLD = 0.5  
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
        _chroma_client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))
        
    try:
        return _chroma_client.get_collection(name="ragnarok_knowledge_base")
    except Exception:
        return None

def retrieve_context(query: str, top_k: int = 3) -> list:
    collection = _get_collection()
    # Passo A: Transformar a query (string) em embedding (lista de floats)
    embed_model = _get_embed_model()
    query_embedding = embed_model.encode(query).tolist()
    # Passo B: Buscar os vetores mais próximos no banco
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]
    recuperados = []    
    # Passo C: Montar o dicionário de retorno e aplicar o Refuse
    for i in range(len(documents)):
        distancia = distances[i]       
        chunk_data = {
            "id": ids[i],
            "texto": documents[i],
            "metadata": metadatas[i],
            "score_distancia": round(distancia, 4)
        }
        # LÓGICA DE REFUSE:
        if distancia > REFUSE_THRESHOLD:
            chunk_data["evidencia_suficiente"] = False
            chunk_data["refuse_motivo"] = f"Distância {distancia:.2f} excedeu limiar {REFUSE_THRESHOLD}"
        else:
            chunk_data["evidencia_suficiente"] = True
        recuperados.append(chunk_data)
    return recuperados