
# Esta tarefa converte os chunks de texto (saída da Tarefa 02) em vetores
# numéricos densos (embeddings) utilizando o modelo Sentence-Transformer
# do BERT-base. A arquitetura funciona assim:
#
#   1. Tokenização: texto → tokens (subword/WordPiece)
#   2. Transformer L6: 6 camadas de self-attention + feed-forward
#   3. Pooling: média dos tokens de saída → vetor 384-d
#   4. Normalização: L2-normalização (vetor unitário)
#
# A dimensão 384 é um bom balanço entre capacidade expressiva e custo
# computacional. O "v2" indica melhorias no treinamento contrastivo
# (siamese networks com triplet loss / cosine similarity loss).

from model_cache import get_sentence_transformer

# CONFIGURAÇÃO

MODEL_NAME = "all-MiniLM-L6-V2"

# FUNÇÃO PRINCIPAL

def generate_embeddings(chunks_list: list) -> list:
    """Gera vetores de embeddings para cada chunk de texto processado.

    A função percorre a lista de chunks, extrai o texto de cada um (chave
    ``"texto"``, conforme gerado pela Tarefa 02), e utiliza o modelo
    Sentence-Transformer ``all-MiniLM-L6-v2`` para converter cada texto em
    um vetor numérico de 384 dimensões.

    O embedding gerado é um **vetor denso** que representa o significado
    semântico do texto. Quanto mais próximo dois embeddings estiverem no
    espaço 384-dimensional (medido por cosseno ou dot product), mais
    semanticamente relacionados são os textos originais. Isto permite:

      • Busca semântica (Tarefa 05): encontrar chunks relevantes para
        uma pergunta mesmo sem correspondência exata de palavras.
      • Clustering: agrupar chunks por tópico.
      • Classificação: categorizar automaticamente o conteúdo.

    Parâmetros:
        chunks_list (list): Lista de dicionários (chunks) gerada na
            Tarefa 02 (chunking.py). Cada dicionário deve conter pelo
            menos a chave ``"texto"`` com o conteúdo textual do chunk.

    Retorno:
        list: A mesma lista recebida, mas com a chave ``"embedding"``
        adicionada a cada dicionário. O valor é uma lista de floats
        de 384 posições representando o vetor de embedding.

    Levanta:
        TypeError: Se chunks_list não for uma lista ou se algum item
            não for um dicionário.
        KeyError: Se um chunk não contiver a chave ``"texto"``.
        RuntimeError: Se o modelo não puder ser carregado.

    Exemplo de saída de um chunk processado:

    .. code-block:: python

        {
            "id": "doc_abc123_chunk_0001",
            "doc_id": "doc_abc123",
            "texto": "Procedimento de recuperação de acesso ao sistema...",
            "metadata": {...},
            "embedding": [0.0234, -0.1567, 0.0892, ..., 0.0123]  # 384 floats
        }
    """
    # Validação de entrada (defensive programming)

    if not isinstance(chunks_list, list):
        raise TypeError(
            f"chunks_list deve ser uma lista, recebeu {type(chunks_list).__name__}."
        )

    if not chunks_list:
        return []

    # Extração dos textos - valida estrutura de cada chunk

    textos = []
    for i, chunk in enumerate(chunks_list):
        if not isinstance(chunk, dict):
            raise TypeError(
                f"Item na posição {i} não é um dicionário, "
                f"recebeu {type(chunk).__name__}."
            )
        if "texto" not in chunk:
            raise KeyError(
                f"Chunk na posição {i} não possui a chave 'texto'. "
                f"Chaves encontradas: {list(chunk.keys())}"
            )
        textos.append(chunk["texto"])

    # Carregamento do modelo (cache compartilhado com chunking.py)

    try:
        model = get_sentence_transformer()
    except Exception as exc:
        raise RuntimeError(
            f"Não foi possível carregar o modelo '{MODEL_NAME}'. "
            "Verifique sua conexão com a internet para o download inicial."
        ) from exc

    # O método encode() do SentenceTransformer processa em batch e
    # utiliza aceleração GPU se disponível (CUDA / MPS).
    # show_progress_bar=False mantém a saída limpa em produção; mude
    # para True para depuração com muitos chunks.
    embeddings = model.encode(
        textos,
        show_progress_bar=False,
        batch_size=32,  # Ajustável; 32 é um bom padrão para CPU/GPU
    )

    # Acoplar embeddings aos chunks originais

    # embeddings retorna um np.ndarray de shape (n_chunks, 384).
    # Convertemos para lista de floats com .tolist() para que seja
    # serializável em JSON (importante para persistência no banco vetorial).
    for i, embedding in enumerate(embeddings):
        chunks_list[i]["embedding"] = embedding.tolist()

    return chunks_list


# TESTE ISOLADO DA TAREFA 03

if __name__ == "__main__":
    print("═" * 70)
    print("  TESTE ISOLADO DA TAREFA 03 — GERAÇÃO DE EMBEDDINGS")
    print("═" * 70)
    print()
    print("  Simula a saída da Tarefa 02 (chunking.py) e gera embeddings")
    print("  para cada chunk usando all-MiniLM-L6-v2 (384 dimensões).")
    print()

    # ── Chunks de exemplo simulando a saída da Tarefa 02 ──

    chunks_teste = [
        {
            "id": "doc_exemplo_chunk_0001",
            "doc_id": "doc_exemplo",
            "texto": "Procedimento de recuperação de acesso ao sistema institucional. "
                     "Para iniciar a recuperação, o usuário deve acessar a página "
                     "oficial de autenticação e selecionar a opção Esqueci minha senha.",
            "metadata": {
                "titulo": "Procedimento de recuperação de acesso",
                "fonte": "Manual interno - acesso institucional",
            },
        },
        {
            "id": "doc_exemplo_chunk_0002",
            "doc_id": "doc_exemplo",
            "texto": "Depois da redefinição, a nova senha deve respeitar os critérios "
                     "exibidos na tela, incluindo tamanho mínimo e combinação de letras, "
                     "números e caracteres especiais.",
            "metadata": {
                "titulo": "Procedimento de recuperação de acesso",
                "fonte": "Manual interno - acesso institucional",
            },
        },
        {
            "id": "doc_exemplo_chunk_0003",
            "doc_id": "doc_exemplo",
            "texto": "Para iniciar a recuperação, o usuário deve acessar a página "
                     "oficial de autenticação e selecionar a opção Esqueci minha senha. "
                     "Em seguida, deve informar o e-mail institucional cadastrado.",
            "metadata": {
                "titulo": "Procedimento de recuperação de acesso",
                "fonte": "Manual interno - acesso institucional",
            },
        },
    ]

    print(f"  📦 Chunks de teste: {len(chunks_teste)} chunks")
    print()

    resultado = generate_embeddings(chunks_teste)

    print(f"  ✅ Chunks processados: {len(resultado)}")
    print()

    for chunk in resultado:
        print(f"  ┌─ ID: {chunk['id']}")
        texto_preview = chunk["texto"][:70].replace("\n", " ")
        print(f"  ├─ Texto: \"{texto_preview}...\"")
        emb = chunk["embedding"]
        dimensoes = len(emb)
        print(f"  ├─ Embedding: {dimensoes} dimensões")

        # Mostra os 5 primeiros valores com 4 casas decimais
        preview_emb = ", ".join(f"{v:+.4f}" for v in emb[:5])
        print(f"  ├─ Primeiros valores: [{preview_emb}, ...]")

        # Calcula a norma L2 — deve ser aproximadamente 1.0 (normalização)
        norma = sum(x ** 2 for x in emb) ** 0.5
        print(f"  └─ Norma L2: {norma:.6f}  (esperado ≈ 1.0)")
        print()

    print("═" * 70)
    print("  TESTE CONCLUÍDO COM SUCESSO!")
    print("═" * 70)
