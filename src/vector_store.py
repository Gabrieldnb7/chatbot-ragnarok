# Tarefa 04: Armazenamento vetorial

import chromadb
from chromadb.config import Settings

def store_in_vector_db(embedded_chunks: list) -> bool:
    """
    Armazena os chunks, embeddings e metadados no ChromaDB para busca semântica.
    
    Parâmetros:
        embedded_chunks (list): Lista de dicionários, onde cada dicionário contém:
                                - 'id': Identificador único do chunk
                                - 'texto': O texto original do chunk (conforme Tarefas 02 e 03)
                                - 'embedding': A lista de floats representando o vetor
                                - 'metadata': Dicionário com a fonte, página, etc.
                            
    Retorno:
        bool: True se a persistência for bem-sucedida, False caso contrário.
    """
    try:
        # Inicializa o ChromaDB com persistência em disco (evitando perda de dados) 
        # e desativa o envio de telemetria padrão para não expor a base.
        client = chromadb.PersistentClient(
            path="../data/vector_db/chroma_data",
            settings=Settings(anonymized_telemetry=False)    
        )

        # Cria ou recupera a coleção vetorial configurando o algoritmo HNSW para usar
        # similaridade de cosseno, o que otimiza a busca para o modelo SentenceTransformer.
        collection = client.get_or_create_collection(
            name="ragnarok_knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        # Prepara os dados para inserção em lote (batch), extraindo os dicionários e
        # agrupando IDs, vetores, textos e metadados em listas contínuas na memória.
        for idx, chunk in enumerate(embedded_chunks):
            doc_id = chunk.get("id", f"doc_chunk_{idx}")
            
            ids.append(doc_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk["texto"])
            
            metadata = chunk.get("metadata", {"source": "documento_desconhecido"})
            metadatas.append(metadata)

        # Consolida a indexação: salva textos e metadados no SQLite e injeta os vetores
        # no grafo vetorial HNSW para busca semântica. IDs duplicados são rejeitados.
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        return True

    except Exception as e:
        print(f"Erro ao persistir no banco vetorial: {e}")
        return False