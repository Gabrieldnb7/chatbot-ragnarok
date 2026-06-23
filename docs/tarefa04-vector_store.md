# Tarefa 04 — Armazenamento Vetorial (Vector Store)

**Responsável:** @jhhoker

## 📋 Visão Geral

Esta tarefa recebe os **chunks já vetorizados** (com embeddings) produzidos pela Tarefa 03 e os persiste em um **Banco de Dados Vetorial** (ChromaDB). O objetivo é criar o índice semântico que permitirá buscas baseadas em contexto durante a fase de recuperação de informações (Retrieval) do nosso pipeline.

## 🔗 Contrato de Interface

### Entrada

`store_in_vector_db(embedded_chunks: list) -> bool`

**Parâmetro:** `embedded_chunks` — lista de dicionários enriquecida pela Tarefa 03.

Cada dicionário recebido deve ter esta estrutura:

| Chave       | Tipo        | Descrição                                    |
|-------------|-------------|----------------------------------------------|
| `id`        | str         | Identificador único do chunk                 |
| `texto`     | str         | Conteúdo textual original                    |
| `metadata`  | dict        | Metadados (fonte, página, título)            |
| `embedding` | list[float] | Vetor denso gerado pelo Sentence-Transformer |

### Saída

Um valor **Booleano** indicando o resultado da operação de persistência.

* `True` — Se todos os chunks foram indexados com sucesso.
* `False` — Em caso de falha crítica na inicialização do banco ou inserção.

### Validações (defensive programming)

* Se `embedded_chunks` não for uma **lista** → `TypeError`
* Se a lista for vazia → retorna `False` ou aborta silenciosamente.
* Se faltar a chave `"embedding"` em algum chunk → `KeyError`
* Se houver colisão de IDs no banco → Atualiza (Upsert) ou ignora (conforme política de repetição).

---

## 🧠 Fundamentação Teórica

### O que é um Banco de Dados Vetorial?

Bancos relacionais (como PostgreSQL) ou baseados em documentos (como MongoDB) são construídos para buscar **correspondências exatas** (ex: `WHERE palavra = 'senha'`). Um Banco de Dados Vetorial é projetado para buscar **proximidade geométrica**. 

Eles não buscam palavras, buscam coordenadas matemáticas no espaço n-dimensional (neste caso, 384 dimensões). 

| Característica        | Banco Relacional (SQL) | Banco Vetorial (Chroma/FAISS) |
|-----------------------|------------------------|-------------------------------|
| Tipo de Busca         | Exata (Keyword/Filtro) | Aproximada (Semântica/Espacial)|
| Estrutura de Índice   | B-Tree / Hash          | Grafos (HNSW) / Inverted File |
| Medida de Sucesso     | `True/False`           | Grau de similaridade (%)      |
| "gato" e "felino"     | Não se encontram       | Retornam como próximos        |

### O Motor de Busca: HNSW (Hierarchical Navigable Small World)

Para buscar vetores rapidamente em milhões de registros sem precisar comparar um por um (força bruta), o ChromaDB utiliza o algoritmo de busca aproximada **HNSW** (Approximate Nearest Neighbor - ANN).

**Como funciona:**
Imagine um sistema de rodovias. O HNSW constrói um **grafo de múltiplas camadas**. As camadas superiores têm poucas conexões (como rodovias interestaduais que cobrem grandes distâncias rapidamente). Conforme a busca desce pelas camadas, as conexões ficam mais densas (como ruas de um bairro). 
Quando você faz uma pergunta ao chatbot, o HNSW "pula" rapidamente pelas rodovias até chegar à vizinhança semântica da sua pergunta e, então, explora as ruas locais para encontrar os chunks mais próximos.

### A Métrica de Distância: Similaridade de Cosseno

Para saber se dois vetores são "próximos", precisamos de uma fita métrica matemática. Como nossos vetores da Tarefa 03 são L2-normalizados, a melhor métrica é a **Similaridade de Cosseno** (Cosine Similarity). 

Em vez de medir a distância em linha reta entre as pontas dos vetores (Distância Euclidiana), ela mede o **ângulo** entre eles. 

$$\text{Cosine Similarity} (A, B) = \frac{A \cdot B}{\|A\| \|B\|}$$

* **1.0** = Ângulo de 0° (Textos com o exato mesmo significado).
* **0.0** = Ângulo de 90° (Textos totalmente ortogonais/sem relação).
* **-1.0** = Ângulo de 180° (Textos com significados diametralmente opostos).

### Arquitetura de Armazenamento SoA (Structure of Arrays)

Diferente do Python que gerencia listas de dicionários, os motores C++ dos bancos vetoriais processam matrizes. Por isso, a inserção não ocorre chunk a chunk, mas sim por meio da **transposição** dos dados em listas paralelas (Batch Insertion).

```
[ {id:1, text:A, emb:[...]}, {id:2, text:B, emb:[...]} ]
    │
    ▼ Transposição
    │
ids        = [1, 2]
documents  = [A, B]
embeddings = [[...], [...]]
```

Isso satura o barramento de memória da CPU de forma mais eficiente, reduzindo o tempo de I/O na gravação do índice.

---

## ⚙️ Dependências

Adicionadas ao `requirements.txt`:

```
chromadb>=0.4.0
```

Dependências encapsuladas pelo Chroma:

- `sqlite3` — Para armazenar os metadados e documentos puros (para filtragem pré/pós busca).
- `hnswlib` — Biblioteca C++ rápida que gerencia a topologia do grafo de vetores.