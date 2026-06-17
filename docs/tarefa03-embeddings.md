# Tarefa 03 — Geração de Embeddings

**Responsável:** @ocnaibill

## 📋 Visão Geral

Esta tarefa converte os **chunks de texto** produzidos pela Tarefa 02 em **vetores numéricos densos** (embeddings) usando o modelo `all-MiniLM-L6-v2` da biblioteca `sentence-transformers`.

## 🔗 Contrato de Interface

### Entrada

```
generate_embeddings(chunks_list: list) -> list
```

**Parâmetro:** `chunks_list` — lista de dicionários gerada pela Tarefa 02.

Cada dicionário deve ter esta estrutura:

| Chave       | Tipo   | Descrição                              |
|-------------|--------|----------------------------------------|
| `id`        | str    | Identificador único do chunk           |
| `doc_id`    | str    | Identificador do documento original    |
| `texto`     | str    | Conteúdo textual do chunk              |
| `metadata`  | dict   | Metadados do documento (título, fonte) |

### Saída

A **mesma lista**, com a chave `"embedding"` adicionada em cada dicionário:

| Chave       | Tipo          | Descrição                                   |
|-------------|---------------|---------------------------------------------|
| `embedding` | list[float]   | Vetor denso de 384 floats (L2-normalizado)  |

```
{
    "id": "doc_abc123_chunk_0001",
    "doc_id": "doc_abc123",
    "texto": "Procedimento de recuperação de acesso...",
    "metadata": {"titulo": "...", "fonte": "..."},
    "embedding": [0.0174, -0.0175, -0.1271, ..., 0.0123]  # ← NOVO
}
```

### Validações (defensive programming)

- Se `chunks_list` não for uma **lista** → `TypeError`
- Se algum item não for **dicionário** → `TypeError`
- Se faltar a chave `"texto"` em algum chunk → `KeyError`
- Se não for possível carregar o modelo → `RuntimeError`
- Se a lista for vazia → retorna `[]`

## 🧠 Fundamentação Teórica

### O que são Embeddings?

Embeddings são **representações vetoriais densas** que capturam o significado semântico de um texto. Diferente de representações tradicionais como **Bag-of-Words** ou **TF-IDF**:

| Característica        | Bag-of-Words / TF-IDF | Embeddings densos       |
|-----------------------|-----------------------|--------------------------|
| Dimensionalidade      | Milhares (vocabulário)| Centenas (ex: 384)      |
| Representação         | Esparsa (muitos zeros)| Densa (tudo preenchido) |
| Captura semântica     | ❌ (só frequência)    | ✅ (similaridade)       |
| "gato" ≈ "felino"     | ❌                    | ✅ (próximos no espaço) |
| Contexto              | ❌                    | ✅ (BERT/Transformer)   |

### O modelo: all-MiniLM-L6-v2

```
all-MiniLM-L6-v2
├── all      → modelo multilíngue (funciona com português)
├── MiniLM   → arquitetura destilada a partir do BERT-base
├── L6       → 6 camadas Transformer (vs. 12 do BERT-base)
├── v2       → treinamento contrastivo aprimorado
└── 384      → dimensão do embedding de saída
```

**Knowledge Distillation:** O MiniLM aprende imitando um modelo maior (teacher BERT), comprimindo o conhecimento em menos parâmetros. É como resumir um livro grosso mantendo as ideias principais.

### Pipeline de geração

```
Texto: "Procedimento de recuperação de acesso..."
    │
    ▼
[Tokenização]  → subword tokens (WordPiece)
    │
    ▼
[Transformer × 6 camadas] → self-attention + feed-forward
    │
    ▼
[Pooling (mean)] → média de todos os tokens de saída
    │
    ▼
[L2 Normalização] → vetor unitário (norma = 1.0)
    │
    ▼
Embedding: [0.0174, -0.0175, ..., 0.0123]  ← 384 floats
```

### Por que 384 dimensões?

A dimensão 384 é um **trade-off** entre:

- **Capacidade expressiva:** Dimensões maiores capturam mais nuances semânticas
- **Eficiência computacional:** Vetores menores ocupam menos espaço e são mais rápidos para comparar (busca por similaridade)
- **Custo de armazenamento:** 384 floats × 4 bytes ≈ 1,5 KB por chunk

Para comparação:
- BERT-base: 768 dimensões
- OpenAI ada-002: 1536 dimensões
- **all-MiniLM-L6-v2: 384 dimensões** → melhor custo-benefício para protótipos

## ⚙️ Dependências

Adicionadas ao `requirements.txt`:

```
sentence-transformers>=5.0.0
```

Dependências que já estavam no ambiente (vem junto com sentence-transformers):

- `torch` (PyTorch) — motor de inferência
- `transformers` (HuggingFace) — arquitetura do modelo
- `numpy` — manipulação de arrays

## 🚀 Como executar

```bash
# Ativar ambiente
conda activate ragnarok

# Teste isolado da Tarefa 03
python src/embeddings.py

# Pipeline completo (Task 01 + 02 + 03)
python -c "
from src.ingestion import ingest_and_anonymize
from src.chunking import chunk_document
from src.embeddings import generate_embeddings

# Pipeline
texto_limpo = ingest_and_anonymize(arquivo_bruto)
chunks = chunk_document(texto_limpo, metadata)
resultado = generate_embeddings(chunks)
"
```

> **Nota:** Na primeira execução, o modelo (~80 MB) é baixado automaticamente do HuggingFace Hub para o cache local em `~/.cache/huggingface/hub/`.

## 🔄 O que vem depois?

A Tarefa 03 alimenta a **Tarefa 04** (`vector_store.py`), que armazenará os embeddings no banco vetorial (ChromaDB ou FAISS) para posterior busca semântica.

## 🧪 Casos de borda cobertos

| Caso                          | Comportamento                     |
|-------------------------------|-----------------------------------|
| Lista vazia `[]`              | Retorna `[]`                      |
| Chunk sem chave `"texto"`     | `KeyError` com diagnóstico        |
| Item não-dicionário           | `TypeError` com posição           |
| Modelo não carregou           | `RuntimeError`                    |
| Texto vazio (`""`)            | Gera embedding normalmente (zero) |
| Textos longos (>256 tokens)   | Modelo faz truncamento automático |
