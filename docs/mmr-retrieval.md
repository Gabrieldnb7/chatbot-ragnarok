# Issue #6 — Resolução Completa (MMR + Reingestão Limpa + Contexto Expandido)

## O Problema

O método `retrieve_context()` estava retornando resultados altamente redundantes para perguntas amplas (ex: _"Quais são os requisitos do PGD?"_). Os primeiros resultados eram cópias exatas do mesmo parágrafo, da mesma fonte, com o mesmo score.

A investigação revelou **três causas raiz**:
1. **Sujeira na Base Vetorial**: O `app.py` anterior permitia fazer uploads repetidos sem limpar a base de dados. Resultado: o mesmo PDF de 65 chunks virou 325 chunks repetidos. A base estava com 1094 chunks, dos quais **65% eram lixo duplicado**.
2. **Falta de Diversificação**: O algoritmo usava similaridade de cosseno simples (`Cosine`), que tende a buscar apenas a vizinhança mais próxima, ignorando a redundância temática.
3. **Contexto Estreito (`top_k=5`)**: Retornar apenas 5 chunks (média de 700 caracteres cada) limitava o LLM a um escopo de ~800 tokens, o que é muito pouco para respostas abrangentes em modelos modernos.

## O Que Foi Feito

A resolução foi feita em três frentes principais:

### 1. Limpeza e Reingestão da Base de Conhecimento
Criamos o script utilitário `scripts/reingest_pdfs.py` para recriar a base vetorial do zero.
- **Antes**: 1094 chunks (22 ingestões bagunçadas).
- **Depois**: 396 chunks (6 ingestões únicas e limpas, representando exatamente todo o texto útil dos 6 PDFs).
- *Nota técnica*: O algoritmo de chunking foi validado — 396 chunks representam os 197.305 caracteres físicos contidos nos PDFs. Não há perda de dados.

### 2. Implementação do MMR (Maximal Marginal Relevance)
Modificamos o `src/retrieval.py` para usar MMR na seleção final dos chunks.
A fórmula do MMR balanceia Relevância vs. Diversidade:
`MMR = (λ * sim_cosseno(query, chunk)) - ((1-λ) * max_sim_cosseno(chunk, chunks_selecionados))`

- Usamos um pool de candidatos maior (`MMR_CANDIDATES = 15`, que escalou para `30` dependendo do `top_k`) antes de selecionar o set final.
- O parâmetro `λ = 0.65` foi definido empiricamente para priorizar relevância mas penalizar repetição.
- *Otimização extra*: Substituímos a instância pesada e repetitiva do `SentenceTransformer` pelo padrão Singleton usando o nosso `model_cache.py`, poupando ~80MB de RAM.

### 3. Expansão do Contexto para o LLM
Para tirar proveito máximo da diversificação do MMR sem perder os chunks que realmente importam, expandimos o `DEFAULT_TOP_K` no `src/app.py`:
- **Mudança**: de `5` para `10`.
- **Efeito**: Agora o LLM recebe o dobro de contexto (~6.600 caracteres) distribuído entre **múltiplos PDFs ao mesmo tempo** em perguntas amplas, sem redundância entre os parágrafos.

## Benchmark e Testes

Ampliamos o `tests/benchmark_retrieval.py` para suportar simulação com 6 PDFs e introduzimos a métrica de **Cobertura por Fonte**.

Resultados do benchmark (base controlada sintética):
- Cosine puro (similaridade > 0.12): 66.7% de cobertura.
- MMR (λ=0.65): **75.0% de cobertura**. (Melhora de +8.3pp).

Testes na base real com `top_k=10`:
- Para "Quais são os requisitos de implementação do PGD?", o Cosine retornava cópias da mesma fonte. O MMR + Base Limpa agora mapeia **3 PDFs distintos** perfeitamente harmonizados para formar uma resposta robusta.

## Como Reproduzir ou Atualizar no Futuro

Se um novo documento do PGD for aprovado:
1. Coloque o PDF em `data/pdfs/`
2. Execute o script de reingestão limpa:
   ```bash
   conda run -n chatbot python scripts/reingest_pdfs.py
   ```
3. A base será recriada sem duplicatas e pronta para o MMR.

## Arquivos Modificados

| Arquivo | Descrição da Mudança |
|---------|----------------------|
| `src/retrieval.py` | Lógica central do MMR e uso de Singleton do model_cache |
| `src/app.py` | `DEFAULT_TOP_K` atualizado para 10; correção de bug de encoding UTF-8 |
| `tests/benchmark_retrieval.py` | Adicionada simulação de multiplos PDFs e métrica de cobertura |
| `scripts/reingest_pdfs.py` | NOVO: Utilitário para limpar ChromaDB e ingerir a base oficial |
| `docs/mmr-retrieval.md` | Esta documentação |
