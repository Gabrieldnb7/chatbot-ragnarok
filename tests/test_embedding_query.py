"""Teste: palavra solta vs frase completa na busca semântica."""
import sys
sys.path.insert(0, "src")

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.EphemeralClient()
collection = client.get_or_create_collection(
    name="test_words", metadata={"hnsw:space": "cosine"}
)

chunks = [
    {"id": "c1", "texto": "Para recuperar a senha, o usuario deve acessar a pagina de login e clicar em Esqueci minha senha. Um e-mail sera enviado com o link de redefinicao."},
    {"id": "c2", "texto": "O periodo de ferias deve ser solicitado com antecedencia minima de 30 dias. O RH analisa a solicitacao e aprova em ate 5 dias uteis."},
    {"id": "c3", "texto": "A autenticacao em dois fatores adiciona uma camada extra de seguranca. O usuario deve confirmar o codigo enviado ao seu celular cadastrado."},
]
texts = [c["texto"] for c in chunks]
embs = model.encode(texts, normalize_embeddings=True).tolist()
collection.add(ids=[c["id"] for c in chunks], embeddings=embs, documents=texts)

queries = [
    ("senha", "Palavra solta"),
    ("acesso senha", "Duas palavras"),
    ("Como recuperar minha senha?", "Frase completa"),
    ("ferias", "Palavra solta"),
    ("solicitar ferias", "Duas palavras"),
    ("Como solicitar ferias?", "Frase completa"),
]

print(f"{'Query':45s} {'Chunk':6s} {'Score':8s}  {'Trecho'}")
print("-" * 110)
for q_text, q_desc in queries:
    q_emb = model.encode(q_text, normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=1)
    dist = results["distances"][0][0]
    score = 1.0 - (dist / 2.0)
    cid = results["ids"][0][0]
    preview = results["documents"][0][0][:55]
    label = f"{q_desc}: {q_text}"
    print(f"{label:45s} {cid:6s} {score:8.4f}  {preview}")
