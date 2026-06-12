# RAGnarok - Assistente Virtual com RAG

Este projeto consiste em um protótipo de assistente virtual desenvolvido com a arquitetura RAG (Retrieval-Augmented Generation). O sistema é capaz de receber uma pergunta, buscar informações relevantes em uma base de conhecimento própria, e gerar uma resposta clara, fundamentada e com citação das fontes.

## 🎯 Objetivos do Projeto
1. **Ingestão e Anonimização:** Processar documentos removendo dados sensíveis.
2. **Chunking:** Dividir o conteúdo em pedaços menores (chunks) para processamento.
3. **Embeddings:** Converter os textos em vetores numéricos.
4. **Vector Store:** Armazenar os vetores (ChromaDB / FAISS).
5. **Busca Semântica:** Recuperar os chunks mais relevantes baseados na query do usuário.
6. **Orquestração LLM:** Gerar respostas concisas citando as fontes ou acionar "triagem humana" caso haja falta de evidências.
7. **Interface:** Disponibilizar tudo via uma interface web de chat.

## 📁 Estrutura do Projeto

```text
chatbot-ragnarok/
├── data/                    # Base de conhecimento e dados do banco vetorial
│   ├── raw/                 # Documentos brutos (PDF, TXT, FAQs, chamados)
│   ├── processed/           # Textos limpos e anonimizados
│   └── vector_db/           # Armazenamento local do banco (ex: arquivos do ChromaDB)
├── src/                     # Código-fonte principal da aplicação
│   ├── ingestion.py         # Tarefa 01: Ingestão e anonimização
│   ├── chunking.py          # Tarefa 02: Divisão de chunks e metadados
│   ├── embeddings.py        # Tarefa 03: Geração de embeddings
│   ├── vector_store.py      # Tarefa 04: Armazenamento vetorial
│   ├── retrieval.py         # Tarefa 05: Busca semântica e lógica de refuse
│   ├── llm_integration.py   # Tarefa 06: Orquestração do RAG
│   └── app.py               # Tarefa 07: Interface Web (Streamlit/Gradio)
├── requirements.txt         # Dependências do projeto
├── README.md                # Descrição e instruções de execução
└── .env                     # Variáveis de ambiente e chaves de API
```

## 🚀 Como Executar Localmente

### 1. Clonar o Repositório
```bash
git clone https://github.com/Gabrieldnb7/chatbot-ragnarok.git
cd chatbot-ragnarok
```

### 2. Configurar o Ambiente Virtual
Recomendamos o uso do `venv` para isolar as dependências do projeto:
```bash
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate
```

### 3. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 4. Configurar as Variáveis de Ambiente
Crie um arquivo `.env` na raiz do projeto baseado no `.env.example` e adicione as suas chaves de API:
```bash
cp .env.example .env
```
> Edite o arquivo `.env` com a sua API Key do provedor LLM escolhido.

### 5. Executar a Interface
Para subir a aplicação via interface web (exemplo usando Streamlit):
```bash
streamlit run src/app.py
```
