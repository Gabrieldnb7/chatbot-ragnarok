"""Interface Streamlit do chatbot RAGnarok."""

from __future__ import annotations

import importlib
import sys
import uuid
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


def _ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interactions", [])
    st.session_state.setdefault("vector_db", [])
    st.session_state.setdefault("last_ingestion", None)


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

    raise RuntimeError("Tipo de arquivo não suportado. Envie TXT, MD, CSV, LOG ou PDF.")


def _ingest_document(text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not _module_ingest_and_anonymize or not _module_chunk_document or not _module_generate_embeddings or not _module_store_in_vector_db:
        raise RuntimeError("As funções de ingestão ainda não estão disponíveis nos módulos do projeto.")

    cleaned = _module_ingest_and_anonymize(text)
    chunks = _module_chunk_document(cleaned, metadata)
    embedded = _module_generate_embeddings(chunks)
    stored = bool(_module_store_in_vector_db(embedded))

    st.session_state.last_ingestion = {
        "metadata": metadata,
        "cleaned_text": cleaned,
        "chunks_count": len(chunks),
        "stored": stored,
        "doc_id": chunks[0]["doc_id"] if chunks else None,
    }

    return {
        "cleaned_text": cleaned,
        "chunks": embedded,
        "stored": stored,
    }


def _retrieve_context(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if _module_retrieve_context:
        try:
            return _module_retrieve_context(question, top_k=top_k)
        except Exception:
            pass
    if _module_retrieve_relevant_chunks:
        try:
            return _module_retrieve_relevant_chunks(question, top_k=top_k)
        except Exception:
            pass
    return []


def _generate_answer(question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
    if _module_generate_rag_response:
        result = _module_generate_rag_response(question, context)
    elif _module_generate_response:
        result = _module_generate_response(question, context)
    else:
        result = None

    if isinstance(result, dict):
        return result

    if result is not None:
        return {
            "resposta_gerada": str(result),
            "fontes": context,
            "precisou_triagem": False,
            "confianca": None,
        }

    return {
        "resposta_gerada": "Nenhum módulo de busca ou resposta está disponível no momento.",
        "fontes": context,
        "precisou_triagem": True,
        "confianca": 0.0,
    }


def render_chat_interface() -> None:
    _ensure_state()

    st.set_page_config(page_title="Chatbot Ragnarok", page_icon="R", layout="wide")
    st.title("Chatbot Ragnarok")
    st.caption("Interface Streamlit do fluxo RAG.")

    with st.sidebar:
        st.header("Base de conhecimento")
        st.write("Carregue um documento ou cole um texto para ingestão.")

        uploaded_file = st.file_uploader(
            "Documento",
            type=["txt", "md", "csv", "log", "pdf"],
            accept_multiple_files=False,
        )
        document_title = st.text_input("Título do documento", value="Documento carregado")
        document_source = st.text_input("Fonte original", value="upload_local")

        sample_text = st.text_area(
            "Ou cole um texto para ingestão rápida",
            height=160,
            placeholder="Cole aqui o conteúdo bruto do documento...",
        )

        if st.button("Ingerir documento", use_container_width=True):
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
                        "titulo": document_title.strip() or "Documento sem título",
                        "fonte": source_label,
                        "tipo": Path(source_label).suffix.lstrip(".") or "texto",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    result = _ingest_document(raw_text, metadata)
                    st.success(f"Documento ingerido com sucesso. {len(result['chunks'])} chunks indexados.")
            except Exception as exc:
                st.error(f"Falha na ingestão: {exc}")

        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.interactions = []
            st.rerun()

        st.divider()
        st.metric("Chunks indexados", len(st.session_state.get("vector_db", [])))
        if st.session_state.last_ingestion:
            st.caption(f"Última ingestão: {st.session_state.last_ingestion['doc_id']}")

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
                        if st.button("Não gostei", key=f"dislike_{index}", use_container_width=True):
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
                    context = _retrieve_context(question)
                    result = _generate_answer(question, context)

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

