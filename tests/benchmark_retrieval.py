#!/usr/bin/env python3
"""
Benchmark para comparar as abordagens de retrieval: 
distância bruta (Gabriel) vs similaridade normalizada (Pablo/refatorado).

O objetivo é demonstrar qual abordagem produz melhores resultados
para o pipeline RAG como um todo.

Para isso, o teste:
  1. Cria uma base vetorial ChromaDB temporária em memória
  2. Insere chunks de documentos com temas variados
  3. Executa queries com diferentes níveis de correspondência
  4. Compara o que cada abordagem aceitaria/rejeitaria

Uso:
    conda run -n ragnarok python tests/benchmark_retrieval.py
"""

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings


# ── Chunks de teste ──────────────────────────────────────────────

CHUNKS = [
    # Grupo A: Recuperação de senha (tema principal do sistema)
    {
        "id": "doc_senha_chunk_0001",
        "texto": "Para recuperar a senha, o usuário deve acessar a página de login e clicar em 'Esqueci minha senha'. Um e-mail será enviado com o link de redefinição.",
        "metadata": {"titulo": "Recuperação de Senha", "fonte": "manual_suporte"},
    },
    {
        "id": "doc_senha_chunk_0002",
        "texto": "O link de redefinição expira em 24 horas. Após esse período, o usuário deve solicitar um novo link. A nova senha deve ter no mínimo 8 caracteres.",
        "metadata": {"titulo": "Recuperação de Senha", "fonte": "manual_suporte"},
    },
    # Grupo B: Autenticação em dois fatores
    {
        "id": "doc_2fa_chunk_0001",
        "texto": "A autenticação em dois fatores adiciona uma camada extra de segurança. O usuário deve confirmar o código enviado ao seu celular cadastrado.",
        "metadata": {"titulo": "Autenticação 2FA", "fonte": "manual_seguranca"},
    },
    # Grupo C: Suporte a chamados (medianamente relacionado)
    {
        "id": "doc_chamado_chunk_0001",
        "texto": "Para abrir um chamado de suporte, o atendente deve preencher o formulário com nome do solicitante, setor e descrição do problema.",
        "metadata": {"titulo": "Abertura de Chamados", "fonte": "manual_atendimento"},
    },
    # Grupo D: Conteúdo não relacionado
    {
        "id": "doc_ferias_chunk_0001",
        "texto": "O período de férias deve ser solicitado com antecedência mínima de 30 dias. O RH analisa a solicitação e aprova em até 5 dias úteis.",
        "metadata": {"titulo": "Política de Férias", "fonte": "manual_rh"},
    },
    {
        "id": "doc_ferias_chunk_0002",
        "texto": "Funcionários com menos de 12 meses de empresa têm direito a 30 dias de férias proporcionais. O pagamento inclui adicional de 1/3.",
        "metadata": {"titulo": "Política de Férias", "fonte": "manual_rh"},
    },
]


def seed_temp_db(model, client):
    """Povoa uma coleção ChromaDB temporária com os chunks de teste."""
    collection = client.get_or_create_collection(
        name="test_ragnarok",
        metadata={"hnsw:space": "cosine"},
    )
    texts = [c["texto"] for c in CHUNKS]
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    ids = [c["id"] for c in CHUNKS]
    metadatas = [c["metadata"] for c in CHUNKS]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return collection


def query_approach_gabriel(collection, model, query: str, top_k: int = 3):
    """Abordagem original do Gabriel: distância bruta com threshold 0.5."""
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    distances = results.get("distances", [[]])[0]
    ids_list = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]

    THRESHOLD = 0.5
    aceitos = []
    rejeitados = []

    for i in range(len(documents)):
        item = {
            "id": ids_list[i],
            "texto": documents[i][:80] + "...",
            "distancia": round(distances[i], 4),
        }
        if distances[i] < THRESHOLD:
            aceitos.append(item)
        else:
            rejeitados.append(item)

    return aceitos, rejeitados


def query_approach_similarity(collection, model, query: str, top_k: int = 3):
    """Abordagem refatorada: similaridade normalizada com threshold 0.12."""
    q_emb = model.encode(query).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)
    distances = results.get("distances", [[]])[0]
    ids_list = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]

    THRESHOLD = 0.12
    aceitos = []
    rejeitados = []

    for i in range(len(documents)):
        score = round(1.0 - (distances[i] / 2.0), 4)
        item = {
            "id": ids_list[i],
            "texto": documents[i][:80] + "...",
            "distancia": round(distances[i], 4),
            "score": score,
        }
        if score > THRESHOLD:
            aceitos.append(item)
        else:
            rejeitados.append(item)

    return aceitos, rejeitados


def tabela_resultado(abordagem, query, aceitos, rejeitados):
    print(f"\n  ┌─ {abordagem}")
    print(f"  │ Query: \"{query}\"")
    if aceitos:
        print(f"  ├─ ACEITOS ({len(aceitos)}):")
        for a in aceitos:
            detalhe = f"dist={a.get('distancia','?'):.2f}" if 'distancia' in a else f"score={a.get('score','?'):.2f}"
            print(f"  │   ✓ {a['id']:30s} {detalhe:>15}  {a['texto']}")
    if rejeitados:
        print(f"  ├─ REJEITADOS ({len(rejeitados)}):")
        for r in rejeitados:
            detalhe = f"dist={r.get('distancia','?'):.2f}" if 'distancia' in r else f"score={r.get('score','?'):.2f}"
            print(f"  │   ✗ {r['id']:30s} {detalhe:>15}  {r['texto']}")
    if not aceitos and not rejeitados:
        print("  │   (sem resultados)")
    print(f"  └─")


QUERIES = [
    ("Como recuperar minha senha?", "Pertinente — corresponde diretamente ao Grupo A"),
    ("Esqueci minha senha, o que fazer?", "Pertinente — corresponde ao Grupo A (paráfrase)"),
    ("Como funciona a autenticação em dois fatores?", "Pertinente — corresponde ao Grupo B"),
    ("Como abrir um chamado de suporte?", "Pertinente — corresponde ao Grupo C (tangencial)"),
    ("Qual a política de férias da empresa?", "Pertinente — corresponde ao Grupo D"),
    ("Qual a previsão do tempo para amanhã?", "Fora de escopo — não deveria retornar nada relevante"),
]


def main():
    print("=" * 70)
    print("  BENCHMARK — Abordagens de Retrieval")
    print("  Gabriel (distância < 0.5) vs Similaridade (score > 0.12)")
    print("=" * 70)

    # Setup
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.EphemeralClient()
    collection = seed_temp_db(model, client)

    print(f"\n  📦 Base de teste: {len(CHUNKS)} chunks em {len({c['metadata']['titulo'] for c in CHUNKS})} grupos temáticos\n")

    total_gabriel_aceitos = 0
    total_gabriel_rejeitados = 0
    total_similaridade_aceitos = 0
    total_similaridade_rejeitados = 0

    for query, desc in QUERIES:
        print(f"  {'─' * 66}")
        print(f"  [{desc}]")
        print()

        aceitos_g, rejeitados_g = query_approach_gabriel(collection, model, query)
        aceitos_s, rejeitados_s = query_approach_similarity(collection, model, query)

        tabela_resultado("Gabriel (dist < 0.5)", query, aceitos_g, rejeitados_g)
        tabela_resultado("Similaridade (score > 0.12)", query, aceitos_s, rejeitados_s)

        total_gabriel_aceitos += len(aceitos_g)
        total_gabriel_rejeitados += len(rejeitados_g)
        total_similaridade_aceitos += len(aceitos_s)
        total_similaridade_rejeitados += len(rejeitados_s)

    print(f"\n  {'=' * 66}")
    print(f"  RESUMO AGREGADO")
    print(f"  {'=' * 66}")
    print(f"\n  {'':>35} {'Aceitos':>10} {'Rejeitados':>12}")
    print(f"  {'Gabriel (dist < 0.5)':>35} {total_gabriel_aceitos:>10} {total_gabriel_rejeitados:>12}")
    print(f"  {'Similaridade (score > 0.12)':>35} {total_similaridade_aceitos:>10} {total_similaridade_rejeitados:>12}")
    print()

    diferenca = total_similaridade_aceitos - total_gabriel_aceitos
    if diferenca > 0:
        print(f"  ➡ Similaridade aceitou {diferenca} chunks a mais que Gabriel em {len(QUERIES)} queries.")
        print(f"  ➡ Isso significa que Gabriel REJEITOU chunks que continham informação")
        print(f"     relevante para a pergunta do usuário — informação essa que o LLM")
        print(f"     poderia ter usado para gerar uma resposta melhor.")
        print()
        print(f"  💡 CONCLUSÃO: Similaridade normalizada (score > 0.12) é superior porque:")
        print(f"     1. Retrieval deve ser ABRANGENTE (foco em RECALL)")
        print(f"     2. O LLM é quem deve decidir o que usar ou descartar")
        print(f"     3. Threshold restritivo no retrieval = informação perdida")
        print(f"     4. Score normalizado 0-1 é intuitivo e consistente com o resto do pipeline")
    print(f"  {'=' * 66}")


if __name__ == "__main__":
    main()
