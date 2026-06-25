# Revisão de Código — Tarefas 01 e 02

> **Revisor:** @ocnaibill
> **Data:** 17/06/2026
> **Contexto:** O Cotonete (Tarefa 02) pediu revisão do próprio código e também sinalizou suspeitas sobre o código do Felipe (Tarefa 01).

---

## 📄 Tarefa 01 — Felipe (`src/ingestion.py`)

### Problemas Encontrados

#### 🔴 1. Falsos positivos no regex de telefone (GRAVE)

```python
re.sub(r'\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b', ...)
```

O padrão `\d{4,5}[-\s]?\d{4}` captura **qualquer** sequência de 4-5 dígitos seguida de 4 dígitos, incluindo:

- `protocolo 1234-5678` → ❌ Anonimiza como telefone
- `processo 98765-4321` → ❌ Anonimiza como telefone
- `data 2025-1234` → ❌ Anonimiza como telefone

**Impacto:** Dados estruturados importantes (números de protocolo, processo, chamado) são perdidos na anonimização.

**Sugestão:** Exigir pelo menos 2 dígitos de DDD ou prefixo `+55` para considerar telefone.

#### 🔴 2. Falsos positivos no regex de cartão de crédito (GRAVE)

```python
re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[DADO BANCARIO REMOVIDO]', texto_limpo)
```

Qualquer sequência de 16 dígitos agrupados em 4 é anonimizada:

- `protocolo 1234 5678 9012 3456` → ❌ Anonimiza como cartão
- `ID 4321 8765 2109 6543` → ❌ Anonimiza como cartão

**Sugestão:** Usar algoritmo de Luhn para validar se é realmente um cartão de crédito, ou exigir palavras-chave como "cartão", "pagamento" próximas ao número.

#### 🟡 3. Erro silencioso em vez de exceção

```python
if not isinstance(file_content, str):
    return ""   # ← Problema!
```

Se o parâmetro vier com tipo errado, a função retorna string vazia sem alerta. Isso **mascara** erros de integração. O `chunking.py` do Cotonete faz certo (levanta `TypeError`).

**Sugestão:** Substituir por `raise TypeError(...)`.

#### 🟡 4. Modelo spaCy carregado na importação

```python
nlp = spacy.load("pt_core_news_sm")  # Carrega no import do módulo
```

Isso significa que qualquer `import ingestion` já carrega o spaCy (~45 MB, alguns segundos), mesmo que a função `ingest_and_anonymize()` nunca seja chamada.

**Sugestão:** Usar **lazy loading** (variável global `_nlp = None` e carregar dentro da função).

#### 🟡 5. CPF sem pontuação não é capturado

O regex exige pontos e traço (`\d{3}\.\d{3}\.\d{3}-\d{2}`). Um CPF digitado como `12345678900` (11 dígitos contínuos) **não é anonimizado**.

**Sugestão:** Adicionar um segundo regex para CPF sem pontuação.

#### ℹ️ 6. NER do spaCy limitado a PER

Só anonimiza `PER` (pessoas). Pode ser intencional, mas vale considerar se `ORG` (organizações) também deveria ser anonimizado dependendo do contexto do projeto.

---

## 📄 Tarefa 02 — Cotonete (`src/chunking.py`)

### Pontos de Atenção

#### 🟡 1. doc_id derivado apenas dos metadados

```python
def _generate_doc_id(metadata: dict) -> str:
    return f"doc_{hash_metadata[:12]}"
```

Se dois documentos tiverem o mesmo título e fonte (ex: dois chamados de "Recuperação de Senha"), terão o **mesmo doc_id**.

**Sugestão:** Misturar um hash do conteúdo do texto no doc_id para garantir unicidade.

#### 🟡 2. Separador no final do chunk

```python
keep_separator="end"
```

Isso coloca o separador (`\n`, `.`, etc.) no **final** do chunk anterior, não no início do próximo. O chunk 2 pode começar no meio de uma frase. Funciona, mas é uma escolha sutil que afeta quem consome os chunks depois.

**Sugestão:** Avaliar se `keep_separator="start"` não seria mais intuitivo para o pipeline.

### Pontos Positivos ✅

- Validações de tipo claras com mensagens descritivas
- `metadata.copy()` evita mutações compartilhadas entre chunks
- Id incremental no chunk (`_0001`, `_0002`) facilita ordenação
- Código limpo, bem documentado, com docstrings
- Teste isolado com texto sintético representativo
- Uso correto do `RecursiveCharacterTextSplitter` com separadores em português

---

## 📄 Tarefa 03 — @ocnaibill (`src/embeddings.py`)

### Destaques da Implementação

- ✅ **Lazy loading** do modelo (não carrega na importação)
- ✅ Conversão `numpy.ndarray` → `list` (serializável em JSON)
- ✅ Validação posicional (informa exatamente qual chunk deu erro)
- ✅ `RuntimeError` com diagnóstico para falha de download do modelo
- ✅ L2-normalização verificada (normas ≈ 1.0)
- ✅ `sentence-transformers` adicionado ao `requirements.txt`

---

## Resumo das Ações Necessárias

| Prioridade | O quê | Quem |
|------------|-------|------|
| 🔴 Alta | Corrigir falsos positivos telefone | Felipe |
| 🔴 Alta | Corrigir falsos positivos cartão | Felipe |
| 🟡 Média | Erro silencioso → exceção | Felipe |
| 🟡 Média | Lazy loading do spaCy | Felipe |
| 🟡 Média | CPF sem pontuação | Felipe |
| 🟡 Média | doc_id incluir conteúdo | Cotonete |
| 🟢 Baixa | Avaliar keep_separator | Cotonete |
| ✅ Concluída | Tarefa 03 implementada | @ocnaibill |
