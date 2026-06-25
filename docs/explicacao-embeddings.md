# Explicação Detalhada — Tarefa 03: Geração de Embeddings

## Explicação Linha a Linha do `src/embeddings.py`

---

### Cabeçalho (linhas 1-13)

```python
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
```

Aqui é só **comentário** — não é código executado. É documentação explicando o que a gente vai fazer. Esses 4 passos são o pipeline inteiro do modelo, explicados mais abaixo.

---

### Linha 15: Importação

```python
from sentence_transformers import SentenceTransformer
```

**O que faz:** Importa a classe `SentenceTransformer` de uma biblioteca chamada `sentence-transformers`.

**Conceito:** Em Python, `from X import Y` significa "pega o módulo X e extrai dele só o objeto Y para usar aqui". É como importar uma ferramenta específica de uma caixa de ferramentas, em vez de trazer a caixa inteira.

**O que é a biblioteca?** O `sentence-transformers` é uma biblioteca feita pelo time do HuggingFace que pega modelos Transformer (como o BERT) e os adapta especificamente para gerar **embeddings de frases** (sentences). Ela abstrai toda a complexidade de:

- Baixar o modelo da internet
- Carregar os pesos do modelo na memória
- Tokenizar o texto
- Passar pelo Transformer
- Aplicar pooling
- Retornar o vetor

Sem ela, você precisaria de umas 200 linhas de PyTorch manual para fazer o mesmo.

---

### Linha 19: Constante

```python
MODEL_NAME = "all-MiniLM-L6-V2"
```

**O que faz:** Define uma **constante** — uma variável que (por convenção) não muda durante o programa. O valor é o nome do modelo no HuggingFace Hub.

**O que é esse nome?** É um identificador que o `sentence-transformers` usa para baixar o modelo certo do repositório do HuggingFace. Cada parte do nome significa algo:

| Parte   | Significado |
|---------|-------------|
| `all`   | Multilíngue (funciona com português, inglês, etc.) |
| `MiniLM`| Arquitetura: uma versão **miniaturizada** do BERT |
| `L6`    | **6** camadas Transformer (vs. 12 do BERT original) |
| `V2`    | Segunda versão do treinamento (treinamento contrastivo melhorado) |

**Por que não usar o BERT original?** O BERT-base tem 12 camadas e 768 dimensões. O MiniLM-L6 reduz para 6 camadas e 384 dimensões. Isso é **conhecimento destilado** (knowledge distillation) — você treina um modelo pequeno para imitar um modelo grande. O resultado é 4x mais rápido e 80% menor, com ~95% da qualidade. Para um protótipo de faculdade, é ideal.

---

### Linha 24: Variável global de cache

```python
_model = None
```

**O que faz:** Cria uma variável global chamada `_model` com valor inicial `None`.

**Por que o underline?** Convenção Python: `_` na frente significa "variável privada / interna". Não é para ser usada fora deste módulo.

**Por que None?** None é o "nada" do Python. A gente usa None para dizer "ainda não carreguei o modelo". Quando precisarmos dele, a gente carrega e guarda nessa variável. Isso evita carregar o modelo toda vez que a função é chamada — **cache em memória**.

---

### Linhas 27-41: Função auxiliar `_get_model`

```python
def _get_model() -> SentenceTransformer:
    """Carrega e retorna o modelo de embeddings com cache em memória.

    Returns:
        SentenceTransformer: Modelo pronto para gerar embeddings.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model
```

**Linha a linha:**

1. `def _get_model() -> SentenceTransformer:` — Declaração de função. O `-> SentenceTransformer` é uma **type hint** — uma dica de que essa função retorna um objeto do tipo `SentenceTransformer`. Python não obriga, mas ajuda quem lê o código e quem usa ferramentas de autocomplete.
2. Docstring — Documentação dentro da função. Explica o que a função faz e o que retorna.
3. `global _model` — Diz ao Python: "a variável `_model` que vou usar aqui NÃO é local — é a variável global lá de cima". Sem isso, o Python criaria uma variável local separada e a global continuaria None para sempre.
4. `if _model is None:` — Verifica se o modelo ainda não foi carregado.
5. `_model = SentenceTransformer(MODEL_NAME)` — Carrega o modelo. Isso:
   - Verifica se o modelo já está em cache no disco (`~/.cache/huggingface/hub/`)
   - Se não estiver, **baixa da internet** (~80 MB)
   - Carrega os pesos na memória RAM/GPU
   - Retorna um objeto pronto para uso
6. `return _model` — Retorna o modelo (recém-carregado ou já em cache).

**Lazy Loading (carregamento preguiçoso):** O modelo só carrega quando `generate_embeddings()` é chamada de fato. Se você apenas importar o módulo, o modelo não carrega. Isso é diferente do que o Felipe fez com o spaCy (modelo carrega na importação).

---

### Linha 46: Função principal

```python
def generate_embeddings(chunks_list: list) -> list:
```

A função que você precisava implementar. Recebe uma lista e devolve uma lista.

**Type hints:**
- `chunks_list: list` — parâmetro é uma lista
- `-> list` — retorno é uma lista

---

### Linhas 94-100: Validação de entrada

```python
if not isinstance(chunks_list, list):
    raise TypeError(
        f"chunks_list deve ser uma lista, recebeu {type(chunks_list).__name__}."
    )

if not chunks_list:
    return []
```

**Primeira validação:**
- `isinstance(chunks_list, list)` pergunta: "chunks_list é uma instância da classe list?"
- `not isinstance(...)` inverte — "NÃO é uma lista?"
- Se não for, `raise TypeError(...)` interrompe o programa com uma mensagem de erro clara

**Defensive programming:** Se alguém passar `generate_embeddings("string")` sem essa verificação, o erro vai acontecer lá na linha 116 (`for i, chunk in enumerate(chunks_list)`), e a mensagem de erro vai ser algo como `TypeError: 'str' object is not iterable`. Isso não diz nada sobre o problema real. Com a verificação, a mensagem é clara: "você passou uma string, mas eu esperava uma lista".

**Segunda validação (caso de borda):** Se a lista for vazia (`[]`), `not []` é `True`. Retorna lista vazia sem fazer nada. Evita processamento desnecessário.

---

### Linhas 104-116: Extração dos textos

```python
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
```

**Linha a linha:**

1. `textos = []` — Cria uma lista vazia para guardar os textos
2. `for i, chunk in enumerate(chunks_list):`
   - `enumerate()` é uma função embutida do Python que, para cada item da lista, retorna um par `(índice, item)`.
   - Exemplo: se `chunks_list = [{"texto": "abc"}, {"texto": "def"}]`, `enumerate` gera `(0, {"texto": "abc"})`, depois `(1, {"texto": "def"})`.
   - `i` vira o índice (0, 1, 2...) e `chunk` vira o dicionário.
   - **Por que usar enumerate em vez de `for chunk in chunks_list`?** Para saber a posição do chunk que deu erro. Se o chunk na posição 5 estiver mal formatado, a mensagem de erro fala exatamente isso.
3. `if not isinstance(chunk, dict):` — Verifica se cada item da lista é um dicionário
4. `raise TypeError(...)` — Se não for, erro com diagnóstico
5. `if "texto" not in chunk:` — Verifica se o dicionário tem a chave `"texto"`
6. `raise KeyError(...)` — Se não tiver, erro mostrando quais chaves existem
7. `textos.append(chunk["texto"])` — Pega o valor da chave `"texto"` e adiciona na lista `textos`

**Conceito: Dicionários Python**
```python
chunk = {"id": "doc_001", "texto": "algum texto"}
```
Dicionários são pares `chave: valor`. A chave `"texto"` guarda o texto do chunk. Acessamos com `chunk["texto"]`.

**No final do loop:** `textos` é uma lista de strings como:
```python
["Procedimento de recuperação...", "Depois da redefinição...", ...]
```

---

### Linhas 120-126: Carregamento do modelo

```python
try:
    model = _get_model()
except Exception as exc:
    raise RuntimeError(
        f"Não foi possível carregar o modelo '{MODEL_NAME}'. "
        "Verifique sua conexão com a internet para o download inicial."
    ) from exc
```

**Estrutura try/except:**

- `try:` — Tenta executar o bloco
- `model = _get_model()` — Chama a função que carrega o modelo (lazy loading)
- `except Exception as exc:` — Se QUALQUER erro acontecer no bloco try, captura o erro
- `raise RuntimeError(...) from exc` — Levanta um novo erro mais explicativo, **encadeado** ao erro original (isso é o `from exc` — preserva o erro original como "causa")

**Por que fazer isso?** O erro original do SentenceTransformer pode ser algo como `ConnectionError: DNS resolution failed` ou `HTTPError: 404`. A gente captura e transforma em algo que faz sentido pro usuário do nosso código: "Não foi possível carregar o modelo, verifique sua internet."

---

### Linhas 132-136: Geração dos embeddings

```python
embeddings = model.encode(
    textos,
    show_progress_bar=False,
    batch_size=32,
)
```

Aqui é onde a transformação acontece.

**O que é `model.encode()`?** É o método principal do SentenceTransformer. Recebe textos e devolve vetores (embeddings).

**Parâmetros:**

1. **`textos`** — Uma lista de strings. A função vai processar cada string e gerar um vetor para cada uma.
2. **`show_progress_bar=False`** — Mostra ou não uma barra de progresso no terminal. Falso para não poluir a saída.
3. **`batch_size=32`** — Quantos textos processar por vez. Ao invés de processar um de cada vez, o modelo processa 32 de uma vez, aproveitando a vetorização (SIMD/GPU). É como uma padaria que assa 32 pães de uma vez em vez de um por um.

### O que acontece internamente quando `encode()` é chamado?

```
textos: ["Procedimento de...", "Depois da...", "Para iniciar..."]
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                   encode()                                │
│                                                          │
│  Passo 1: Tokenização                                    │
│  ┌─────────────────────────────────────────────┐        │
│  │ "Procedimento de recuperação"               │        │
│  │      ↓                                       │        │
│  │ ["[CLS]", "pro", "##cedi", "##mento",       │        │
│  │  "de", "recu", "##pera", "##ção", "[SEP]"]  │        │
│  └─────────────────────────────────────────────┘        │
│                                                          │
│  Passo 2: Embedding Lookup                               │
│  Cada token vira um vetor de 384 números                │
│  (inicialmente aleatórios, refinados no treinamento)    │
│                                                          │
│  Passo 3: Transformer × 6 camadas                       │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──┐│
│  │Self- │→ │Feed  │→ │Self- │→ │Feed  │→ │Self- │→ │FF││
│  │Attn  │  │Fwd   │  │Attn  │  │Fwd   │  │Attn  │  │  ││
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──┘│
│  Camada 1   Camada 2   ...         Camada 6              │
│                                                          │
│  Cada camada permite que cada token "olhe" para os      │
│  outros tokens (self-attention) e entenda o contexto.   │
│                                                          │
│  Passo 4: Mean Pooling                                   │
│  Tira a MÉDIA dos vetores de todos os tokens             │
│  → um ÚNICO vetor de 384 números para a frase toda      │
│                                                          │
│  Passo 5: L2 Normalização                                │
│  Divide cada número pela norma do vetor inteiro          │
│  → vetor "unitário" (norma = 1.0)                       │
│                                                          │
│  Resultado: np.array de shape (3, 384)                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ [[0.017, -0.017, ..., 0.012],   ← chunk 1       │   │
│  │  [-0.004,  0.017, ..., 0.009],  ← chunk 2       │   │
│  │  [0.010,  0.010, ..., -0.007]]  ← chunk 3       │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

### Self-Attention (explicação intuitiva)

Imagine a frase: "O gato sentou no tapete porque ele estava cansado."

Quem é "ele"? O gato ou o tapete? Um humano sabe que é o gato (gado cansado). Um computador tradicional não.

Self-attention resolve isso: cada palavra **"olha" para todas as outras palavras** e calcula "quanto eu me relaciono com cada uma?". A palavra "ele" aprende a dar mais peso para "gato" e menos para "tapete" com base no contexto.

6 camadas disso significa que esse processo de "olhar para os outros" acontece 6 vezes em sequência, refinando a representação a cada passo.

---

### Mean Pooling

O Transformer gera um vetor para **cada token** da frase. Mas a gente quer UM vetor para a **frase inteira**. Pooling é simplesmente: tira a média de todos os vetores. Um vetor médio que representa o sentido geral.

---

### L2 Normalização

Depois do pooling, a gente **normaliza** o vetor: divide cada componente pela raiz da soma dos quadrados (norma L2). O resultado é um vetor de **tamanho 1** (norma = 1.0, ou perto disso).

Por quê? Para que a **similaridade por cosseno** entre dois embeddings seja simplesmente o produto escalar entre eles — mais eficiente computacionalmente.

---

### Linhas 143-146: Acoplar embeddings aos chunks

```python
for i, embedding in enumerate(embeddings):
    chunks_list[i]["embedding"] = embedding.tolist()

return chunks_list
```

1. `for i, embedding in enumerate(embeddings):` — Percorre a matriz de embeddings. Cada `embedding` é um `numpy.ndarray` de 384 números (tipo `float32` do numpy).
2. `chunks_list[i]["embedding"] = ...` — Acessa o chunk na posição `i` e adiciona uma nova chave `"embedding"` ao dicionário, com o vetor como valor.
3. `embedding.tolist()` — **Por que tolist()?** O numpy usa tipos próprios (`numpy.float32`) que não são serializáveis em JSON. `tolist()` converte o array numpy em uma lista Python comum de floats. Isso é crucial porque a Tarefa 04 (banco vetorial) vai precisar salvar esses embeddings em arquivo.
4. `return chunks_list` — Retorna a mesma lista modificada.

---

### Linhas 149-225: Teste isolado

```python
if __name__ == "__main__":
```

**O que faz?** Este bloco só executa se o arquivo for rodado **diretamente** (`python src/embeddings.py`). Se o arquivo for importado por outro (`from embeddings import generate_embeddings`), esse bloco **não executa**. É uma forma de ter código de teste que não polui quando o módulo é usado como biblioteca.

---

## CONCEITOS FUNDAMENTAIS

### 1. O que é um embedding?

Um **embedding** é a tradução de um texto (algo qualitativo, subjetivo) para um **vetor de números** (algo quantitativo, matemático).

```
"gato"     → [0.2, -0.5, 0.8, 0.1, ...]  (384 números)
"felino"   → [0.19, -0.48, 0.79, 0.12, ...]  (quase igual!)
"bicicleta" → [-0.7, 0.3, -0.1, 0.9, ...]  (muito diferente!)
```

Duas palavras parecidas geram vetores parecidos. Duas palavras diferentes geram vetores distantes. Isso é o que permite **busca semântica** — encontrar "felino" quando o usuário pesquisa "gato", mesmo que a palavra exata não apareça.

### 2. O que é um Transformer?

É a arquitetura de rede neural que revolucionou o NLP em 2017 (Google: "Attention is All You Need"). A ideia central é **self-attention**: cada palavra da frase "olha" para todas as outras para entender o contexto.

Antes do Transformer, modelos liam texto da esquerda para a direita (LSTM/RNN). O problema: palavras no começo da frase "esqueciam" palavras no final. O Transformer resolve isso permitindo que qualquer palavra se conecte com qualquer outra, não importa a distância.

### 3. O que é knowledge distillation?

É a técnica de treinar um modelo **professor** (grande, caro, preciso) e usar ele para treinar um modelo **aluno** (pequeno, rápido, quase tão bom). O aluno tenta imitar as saídas do professor. O MiniLM é um aluno do BERT — mantém ~95% da qualidade com ~25% do tamanho.
