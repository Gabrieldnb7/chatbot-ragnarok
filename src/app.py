"""Interface Streamlit do chatbot RAGnarok.

Este arquivo conecta o fluxo completo de RAG descrito no README/roadmap:
ingestão -> limpeza/anonimização -> chunking -> embeddings -> vector store
-> recuperação -> geração de resposta.

O app prioriza os módulos dedicados em `src/` quando eles existem, mas também
mantém fallbacks seguros para continuar utilizável enquanto o pipeline está
sendo concluído.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _optional_import(module_names: Iterable[str], attr: str):
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        candidate = getattr(module, attr, None)
        if callable(candidate):
            return candidate
    return None


_module_ingest_and_anonymize = _optional_import(["ingestion", "src.ingestion"], "ingest_and_anonymize")
_module_chunk_document = _optional_import(["chunking", "src.chunking"], "chunk_document")
_module_generate_embeddings = _optional_import(["embeddings", "src.embeddings"], "generate_embeddings")
_module_store_in_vector_db = _optional_import(["vector_store", "src.vector_store"], "store_in_vector_db")
_module_retrieve_context = _optional_import(["retrieval", "src.retrieval"], "retrieve_context")
_module_retrieve_relevant_chunks = _optional_import(
    ["retrieval", "src.retrieval"], "retrieve_relevant_chunks"
)
_module_generate_rag_response = _optional_import(
    ["llm_integration", "src.llm_integration"], "generate_rag_response"
)
_module_generate_response = _optional_import(["llm_integration", "src.llm_integration"], "generate_response")


DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_TOP_K = 3
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


@dataclass
class RetrievalResult:
    chunk: Dict[str, Any]
    score: float


def _stable_doc_id(metadata: Dict[str, Any]) -> str:
    payload = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"doc_{digest}"


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def ingest_and_anonymize(file_content: str) -> str:
    """Implementação de fallback para a tarefa 01."""
    if _module_ingest_and_anonymize:
        try:
            return _module_ingest_and_anonymize(file_content)
        except Exception:
            pass

    if not isinstance(file_content, str):
        return ""

    text = _normalize_text(file_content)

    replacements = [
        (r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF REMOVIDO]"),
        (r"\b\d{11}\b", "[CPF REMOVIDO]"),
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL REMOVIDO]"),
        (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[DADO BANCARIO REMOVIDO]"),
        (r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b", "[TELEFONE REMOVIDO]"),
        (r"(?i)\bsenha\s*[:=]\s*\S+", "senha: [DADO REMOVIDO]"),
        (r"(?i)\bpassword\s*[:=]\s*\S+", "password: [DADO REMOVIDO]"),
        (r"(?i)\bchave\s*pix\s*[:=]\s*\S+", "chave pix: [DADO REMOVIDO]"),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    try:
        import spacy  # type: ignore

        try:
            nlp = spacy.load("pt_core_news_sm")
        except Exception:
            nlp = None

        if nlp is not None:
            doc = nlp(text)
            anonymized = text
            for ent in reversed(doc.ents):
                if ent.label_ in {"PER", "PERSON", "ORG", "LOC", "GPE", "MONEY", "DATE"}:
                    anonymized = anonymized[: ent.start_char] + "[DADO REMOVIDO]" + anonymized[ent.end_char :]
            return anonymized
    except Exception:
        pass

    return text


def chunk_document(cleaned_text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Implementação de fallback para a tarefa 02."""
    if _module_chunk_document:
        try:
            return _module_chunk_document(cleaned_text, metadata)
        except Exception:
            pass

    if not isinstance(cleaned_text, str):
        raise TypeError("cleaned_text must be a string")
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary")

    cleaned_text = cleaned_text.strip()
    if not cleaned_text:
        return []

    doc_id = _stable_doc_id(metadata)
    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_index = 1
    while start < len(cleaned_text):
        end = min(len(cleaned_text), start + DEFAULT_CHUNK_SIZE)
        chunk_text = cleaned_text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "id": f"{doc_id}_chunk_{chunk_index:04d}",
                    "doc_id": doc_id,
                    "texto": chunk_text,
                    "metadata": dict(metadata),
                }
            )
            chunk_index += 1
        if end >= len(cleaned_text):
            break
        start = max(end - DEFAULT_CHUNK_OVERLAP, start + 1)
    return chunks


def _fallback_embed_text(text: str) -> Dict[str, float]:
    tokens = re.findall(r"\w+", (text or "").lower())
    counts: Dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def generate_embeddings(chunks_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Implementação de fallback para a tarefa 03."""
    if _module_generate_embeddings:
        try:
            return _module_generate_embeddings(chunks_list)
        except Exception:
            pass

    embedded: List[Dict[str, Any]] = []
    for chunk in chunks_list or []:
        updated = dict(chunk)
        updated["embedding"] = _fallback_embed_text(chunk.get("texto", ""))
        embedded.append(updated)
    return embedded


def store_in_vector_db(embedded_chunks: List[Dict[str, Any]]) -> bool:
    """Implementação de fallback para a tarefa 04."""
    if _module_store_in_vector_db:
        try:
            return bool(_module_store_in_vector_db(embedded_chunks))
        except Exception:
            pass

    st.session_state["vector_db"] = list(embedded_chunks or [])
    st.session_state["vector_db_ready"] = bool(embedded_chunks)
    return bool(embedded_chunks)


def _cosine_similarity_dict(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = sum(v * v for v in a.values()) ** 0.5
    norm_b = sum(v * v for v in b.values()) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve_context(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """Implementação de fallback para a tarefa 05."""
    if _module_retrieve_context:
        try:
            return _module_retrieve_context(query, top_k=top_k)
        except Exception:
            pass
    if _module_retrieve_relevant_chunks:
        try:
            return _module_retrieve_relevant_chunks(query, top_k=top_k)
        except Exception:
            pass

    vector_db = st.session_state.get("vector_db", [])
    query_vector = _fallback_embed_text(query)
    scored: List[RetrievalResult] = []

    for chunk in vector_db:
        score = _cosine_similarity_dict(query_vector, chunk.get("embedding", {}))
        scored.append(RetrievalResult(chunk=chunk, score=score))

    scored.sort(key=lambda item: item.score, reverse=True)

    results: List[Dict[str, Any]] = []
    for item in scored[:top_k]:
        chunk = dict(item.chunk)
        chunk["score"] = round(item.score, 4)
        results.append(chunk)
    return results


def generate_rag_response(query: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Implementação de fallback para a tarefa 06."""
    if _module_generate_rag_response:
        try:
            result = _module_generate_rag_response(query, retrieved_context)
            if isinstance(result, dict):
                return result
            return {
                "resposta_gerada": str(result),
                "fontes": retrieved_context,
                "precisou_triagem": False,
            }
        except Exception:
            pass

    if _module_generate_response:
        try:
            result = _module_generate_response(query, retrieved_context)
            if isinstance(result, dict):
                return result
            return {
                "resposta_gerada": str(result),
                "fontes": retrieved_context,
                "precisou_triagem": False,
            }
        except Exception:
            pass

    if not retrieved_context:
        return {
            "resposta_gerada": (
                "Nao encontrei evidencias suficientes na base de conhecimento para responder com seguranca."
            ),
            "fontes": [],
            "precisou_triagem": True,
        }

    best_score = max((chunk.get("score", 0.0) for chunk in retrieved_context), default=0.0)
    precisou_triagem = best_score < DEFAULT_SCORE_THRESHOLD

    if precisou_triagem:
        resposta = (
            "A evidenca recuperada foi fraca para uma resposta confiavel. "
            "Recomendo triagem humana ou a inclusao de documentos mais especificos."
        )
    else:
        snippets = []
        for idx, chunk in enumerate(retrieved_context, start=1):
            metadata = chunk.get("metadata", {})
            source_label = metadata.get("titulo") or metadata.get("fonte") or chunk.get("id", "fonte_desconhecida")
            snippet = chunk.get("texto", "")[:260].strip()
            snippets.append(f"{idx}. [{source_label}] {snippet}")
        resposta = (
            "Com base nos trechos recuperados, seguem os pontos mais relevantes:\n"
            + "\n".join(snippets)
            + "\n\nSe quiser, posso aprofundar com mais contexto ou buscar outra consulta."
        )

    return {
        "resposta_gerada": resposta,
        "fontes": retrieved_context,
        "precisou_triagem": precisou_triagem,
    }


def _extract_text_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""

    file_name = (uploaded_file.name or "").lower()
    raw_bytes = uploaded_file.getvalue()

    if file_name.endswith((".txt", ".md", ".csv", ".log")):
        return raw_bytes.decode("utf-8", errors="ignore")

    if file_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except Exception:
                raise RuntimeError("PDF support requires pypdf or PyPDF2 to be installed.")

        reader = PdfReader(uploaded_file)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    raise RuntimeError(
        "Tipo de arquivo não suportado. Envie um arquivo TXT, MD, CSV, LOG ou PDF."
    )


def _render_sources(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return
    with st.expander("Fontes consultadas"):
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            titulo = metadata.get("titulo") or chunk.get("id", "Documento")
            fonte = metadata.get("fonte") or metadata.get("source") or "desconhecida"
            score = chunk.get("score")
            texto = chunk.get("texto", "")
            st.markdown(f"**{titulo}**")
            st.caption(f"Fonte: {fonte}" + (f" | score: {score}" if score is not None else ""))
            st.write(texto[:500] + ("..." if len(texto) > 500 else ""))


def _register_feedback(interaction_id: str, value: str) -> None:
    for interaction in st.session_state.interactions:
        if interaction["id"] == interaction_id:
            interaction["feedback_usuario"] = value
            break


def _ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interactions", [])
    st.session_state.setdefault("vector_db", [])
    st.session_state.setdefault("vector_db_ready", False)
    st.session_state.setdefault("last_ingestion", None)


def _ingest_document(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = ingest_and_anonymize(text)
    chunks = chunk_document(cleaned, metadata)
    embedded = generate_embeddings(chunks)
    stored = store_in_vector_db(embedded)

    st.session_state.last_ingestion = {
        "metadata": metadata,
        "cleaned_text": cleaned,
        "chunks_count": len(chunks),
        "stored": stored,
        "doc_id": chunks[0]["doc_id"] if chunks else _stable_doc_id(metadata),
    }

    return {
        "cleaned_text": cleaned,
        "chunks": embedded,
        "stored": stored,
    }


def _run_rag_pipeline(question: str) -> Dict[str, Any]:
    try:
        retrieved_context = retrieve_context(question, top_k=DEFAULT_TOP_K)
        if not retrieved_context:
            return {
                "resposta_gerada": (
                    "Nao encontrei evidencias suficientes na base de conhecimento para responder com seguranca."
                ),
                "fontes": [],
                "precisou_triagem": True,
                "confianca": 0.0,
            }

        result = generate_rag_response(question, retrieved_context)
        if not isinstance(result, dict):
            result = {
                "resposta_gerada": str(result),
                "fontes": retrieved_context,
                "precisou_triagem": False,
            }

        result.setdefault("fontes", retrieved_context)
        result.setdefault("precisou_triagem", False)
        result.setdefault("confianca", None)
        return result
    except Exception as exc:
        return {
            "resposta_gerada": f"Ocorreu um erro ao processar sua pergunta: {exc}",
            "fontes": [],
            "precisou_triagem": True,
            "confianca": None,
        }


def render_chat_interface() -> None:
    _ensure_state()

    st.set_page_config(page_title="Chatbot Ragnarok", page_icon="R", layout="wide")
    st.title("Chatbot Ragnarok")
    st.caption("Fluxo RAG completo: ingestao, limpeza, chunking, busca e resposta.")

    with st.sidebar:
        st.header("Base de conhecimento")
        st.write("Carregue um documento para alimentar o banco vetorial local.")

        uploaded_file = st.file_uploader(
            "Documento",
            type=["txt", "md", "csv", "log", "pdf"],
            accept_multiple_files=False,
        )
        document_title = st.text_input("Titulo do documento", value="Documento carregado")
        document_source = st.text_input("Fonte original", value="upload_local")

        sample_text = st.text_area(
            "Ou cole um texto para ingestão rápida",
            height=160,
            placeholder="Cole aqui o conteudo bruto do documento...",
        )

        ingest_clicked = st.button("Ingerir documento", use_container_width=True)

        if ingest_clicked:
            try:
                raw_text = sample_text.strip()
                source_label = document_source.strip() or "upload_local"
                if uploaded_file is not None:
                    raw_text = _extract_text_from_upload(uploaded_file)
                    source_label = uploaded_file.name or source_label

                if not raw_text:
                    st.warning("Forneça um arquivo ou cole um texto para ingestão.")
                else:
                    metadata = {
                        "titulo": document_title.strip() or "Documento sem titulo",
                        "fonte": source_label,
                        "tipo": Path(source_label).suffix.lstrip(".") or "texto",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    result = _ingest_document(raw_text, metadata)
                    st.success(
                        f"Documento ingerido com sucesso. {len(result['chunks'])} chunks indexados."
                    )
            except Exception as exc:
                st.error(f"Falha na ingestao: {exc}")

        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.interactions = []
            st.rerun()

        st.divider()
        st.metric("Chunks indexados", len(st.session_state.get("vector_db", [])))
        if st.session_state.last_ingestion:
            st.caption(f"Ultima ingestao: {st.session_state.last_ingestion['doc_id']}")

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.subheader("Status")
        st.write(
            "Use a área principal para perguntar sobre os documentos carregados. "
            "Se a evidência for fraca, o app marca triagem humana."
        )
        if st.session_state.last_ingestion:
            st.json(
                {
                    "doc_id": st.session_state.last_ingestion["doc_id"],
                    "chunks": st.session_state.last_ingestion["chunks_count"],
                    "stored": st.session_state.last_ingestion["stored"],
                }
            )
        else:
            st.info("Nenhum documento foi ingerido ainda.")

    with col_main:
        for index, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    _render_sources(message["sources"])

                if message["role"] == "assistant" and message.get("interaction_id"):
                    feedback_cols = st.columns(2)
                    with feedback_cols[0]:
                        if st.button("Gostei", key=f"like_{index}", use_container_width=True):
                            _register_feedback(message["interaction_id"], "positivo")
                    with feedback_cols[1]:
                        if st.button("Nao gostei", key=f"dislike_{index}", use_container_width=True):
                            _register_feedback(message["interaction_id"], "negativo")

        question = st.chat_input("Digite sua pergunta...")

        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question, "sources": None, "interaction_id": None}
            )
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Buscando contexto e gerando resposta..."):
                    result = _run_rag_pipeline(question)

                st.markdown(result["resposta_gerada"])
                if result.get("fontes"):
                    _render_sources(result["fontes"])

            interaction = {
                "id": f"interacao-{uuid.uuid4().hex[:8]}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pergunta": question,
                "contexto_utilizado": [chunk["id"] for chunk in result.get("fontes", [])],
                "resposta_gerada": result["resposta_gerada"],
                "confianca": result.get("confianca"),
                "precisou_triagem": result.get("precisou_triagem", False),
                "feedback_usuario": None,
            }
            st.session_state.interactions.append(interaction)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result["resposta_gerada"],
                    "sources": result.get("fontes", []),
                    "interaction_id": interaction["id"],
                }
            )


if __name__ == "__main__":
    render_chat_interface()
