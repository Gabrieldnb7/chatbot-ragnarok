# Tarefa 05 — Busca Semântica e Recuperação (Retrieval)

**Responsável:** @Gabrieldnb7

## 📋 Visão Geral

Esta tarefa implementa o módulo responsável por interagir com o Banco Vetorial (ChromaDB) para realizar a **busca semântica**. Quando o usuário envia uma pergunta, este módulo converte a pergunta em um vetor, busca os trechos (chunks) matematicamente mais próximos na base de dados e aplica uma **lógica de recusa** baseada na distância vetorial para mitigar alucinações.

## 🔗 Contrato de Interface

### Entrada

```python
retrieve_context(query: str, top_k: int = 3) -> list
```

**Parâmetros:** 
- `query` (str) — A pergunta ou comando em linguagem natural enviado pelo usuário.
- `top_k` (int, opcional) — A quantidade máxima de chunks que o banco deve retornar (padrão é 3).

### Saída

Uma **lista de dicionários** contendo os chunks recuperados, enriquecidos com a métrica de distância e a flag de recusa.

Cada dicionário possui esta estrutura:

| Chave                  | Tipo   | Descrição                                                                 |
|------------------------|--------|---------------------------------------------------------------------------|
| `id`                   | str    | Identificador único do chunk no banco vetorial                            |
| `texto`                | str    | O texto bruto do chunk retornado                                          |
| `metadata`             | dict   | Metadados associados ao chunk (fonte, título, etc)                        |
| `score_distancia`      | float  | Distância vetorial arredondada para 4 casas decimais                      |
| `evidencia_suficiente` | bool   | Flag que indica se o chunk passou no limiar aceitável de similaridade     |
| `refuse_motivo`        | str    | (Opcional) Motivo textual caso a evidência tenha sido considerada fraca   |

Exemplo de retorno:
```json
[
    {
        "id": "doc_123_chunk_0",
        "texto": "As sanções aplicáveis em caso de descumprimento são...",
        "metadata": {"titulo": "IN_20_2025.pdf"},
        "score_distancia": 0.3251,
        "evidencia_suficiente": true
    },
    {
        "id": "doc_890_chunk_5",
        "texto": "Receita de bolo de cenoura com cobertura...",
        "metadata": {"titulo": "Receitas.pdf"},
        "score_distancia": 0.8920,
        "evidencia_suficiente": false,
        "refuse_motivo": "Distância 0.89 excedeu limiar 0.5"
    }
]
```

### Validações (Defensive Programming)

- A inicialização do banco (`chromadb.PersistentClient`) e a carga do modelo de embeddings (`SentenceTransformer`) utilizam **Lazy Initialization (Singleton)**. Isso previne instâncias duplicadas na memória e erros de File Lock no SQLite no sistema Windows.
- Se o banco de dados falhar ao ser lido ou a collection não existir, `_get_collection()` interceptará a exceção de forma segura.
- O limiar de recusa (`REFUSE_THRESHOLD = 0.5`) impede que respostas completamente fora de contexto poluam a geração do LLM.

---

## 🧠 Fundamentação Teórica

### Distância vs Similaridade

Quando a query do usuário entra no modelo de Embedding, ela é transformada num vetor e jogada no mesmo plano dimensional de 384 dimensões que o resto dos nossos documentos. O ChromaDB então avalia a **distância** entre o vetor da pergunta e os vetores dos textos.

No caso do HNSW parametrizado com `cosine`:
- **Maior Similaridade** = **Menor Distância** (próximo de `0.0`)
- **Menor Similaridade** = **Maior Distância** (próximo de `1.0` ou mais)

### A Lógica de Recusa (Refuse Logic / Thresholding)

Em arquiteturas RAG modernas, o calcanhar de aquiles são as perguntas Fora de Escopo (Out of Domain). O KNN / HNSW (Nearest Neighbors) **sempre** retornará o "top K". Se você perguntar "Qual a cor do céu?" em uma base sobre "Leis de Trânsito", o HNSW vai te trazer as 3 leis que matematicamente passaram mais "perto" (mesmo que longe) dessa pergunta.

Ao definir um `REFUSE_THRESHOLD = 0.5`:
1. Avaliamos matematicamente que qualquer vetor com distância maior que `0.5` **não pertence** semanticamente à intenção da query.
2. Injetamos a flag booleana `"evidencia_suficiente": False`.
3. Transferimos a responsabilidade arquitetural (D.T.O.) de cortar o fluxo para o **Orquestrador**.
4. Quando o Orquestrador filtra os chunks falsos, garantimos o fenômeno do **Early Exit**, impedindo a alucinação (LLM Hallucination) e economizando tokens de API.

### Design Pattern: Lazy Initialization

O modelo `SentenceTransformer` demanda uma forte carga na CPU/RAM e nos sub-processos do Python. Instanciar modelos globais ao nível do módulo (importação) força o download e a ocupação da memória imediatamente.

Utilizamos um padrão de **Lazy Loading**:
```python
_embed_model = None
def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model
```
Isso garante que a aplicação consuma recursos **apenas e no exato momento** em que uma query precisa ser vetorizada.

---

## ⚙️ Dependências

As dependências são compartilhadas majoritariamente com as tarefas `03` e `04`:

- `chromadb` — Para recuperação no banco vetorial via query parametrizada.
- `sentence-transformers` — Para gerar o vetor da pergunta "on the fly".

```bash
# As dependências já devem constar no ambiente:
chromadb>=0.4.0
sentence-transformers>=5.0.0
```

---
