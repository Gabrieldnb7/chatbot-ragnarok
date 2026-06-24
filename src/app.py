# Tarefa 07: Interface Web e Integracao do Fluxo
"""Interface Streamlit do chatbot RAGnarok."""

from __future__ import annotations

import html
import importlib
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
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
_module_clear_vector_db = _optional_import(["vector_store", "src.vector_store"], "clear_vector_db")
_module_load_vector_db = _optional_import(["vector_store", "src.vector_store"], "load_vector_db")
_module_get_collection = _optional_import(["vector_store", "src.vector_store"], "get_collection")
_module_search_similar_chunks = _optional_import(["vector_store", "src.vector_store"], "search_similar_chunks")
_module_retrieve_context = _optional_import(["retrieval", "src.retrieval"], "retrieve_context")
_module_retrieve_relevant_chunks = _optional_import(["retrieval", "src.retrieval"], "retrieve_relevant_chunks")
_module_generate_rag_response = _optional_import(["llm_integration", "src.llm_integration"], "generate_rag_response")
_module_generate_response = _optional_import(["llm_integration", "src.llm_integration"], "generate_response")


PDF_PAGE_MARKER_RE = re.compile(r"\n\n\[Pagina (\d+)\]\n")
MAX_INDEXED_CHUNKS = 5000


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #0b0d12; color: #f3f4f6; }
        [data-testid="stSidebar"] { background: #1b1d26; border-right: 1px solid rgba(255, 255, 255, 0.08); }
        [data-testid="stSidebar"] .stButton button,
        .stButton button,
        .stForm button { border-radius: 8px; }
        .rag-panel { background: #11131a; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 14px; padding: 1rem 1.1rem; }
        .rag-title { font-size: 2rem; font-weight: 800; line-height: 1.1; margin-bottom: 0.35rem; }
        .rag-subtitle { color: rgba(243, 244, 246, 0.7); font-size: 0.95rem; margin-bottom: 1rem; }
        .rag-status { background: #0f3a5d; color: #d7eafe; border-radius: 10px; padding: 0.9rem 1rem; margin-top: 0.8rem; }
        .rag-muted { color: rgba(243, 244, 246, 0.72); font-size: 0.92rem; }
        .rag-scroll-box {
            max-height: 360px;
            overflow-y: auto;
            background: #0f131b;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1rem;
            white-space: pre-wrap;
            line-height: 1.65;
        }
        .rag-answer-box {
            max-height: 380px;
            overflow-y: auto;
            background: #10131b;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            white-space: pre-wrap;
            line-height: 1.7;
            font-size: 1.01rem;
        }
        .rag-source-item { margin-bottom: 1rem; }
        .rag-source-index {
            display: inline-block;
            width: 1.8rem;
            height: 1.8rem;
            line-height: 1.8rem;
            border-radius: 999px;
            background: #ff9f1c;
            color: #10131b;
            font-weight: 800;
            text-align: center;
            margin-right: 0.5rem;
        }
        .rag-mini-card {
            background: #0f131b;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interactions", [])
    st.session_state.setdefault("last_ingestion", None)
    st.session_state.setdefault("question_input", "")


def _render_scrollable_text(text: str) -> None:
    safe_text = html.escape(text or "")
    st.markdown(f'<div class="rag-answer-box">{safe_text}</div>', unsafe_allow_html=True)


def _render_sources(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return

    with st.expander(f"Fontes consultadas ({len(chunks)})"):
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata", {})
            titulo = metadata.get("titulo") or "Documento"
            fonte = metadata.get("fonte") or "documento"
            page_number = metadata.get("page_number")
            page_label = metadata.get("page_label")
            score = chunk.get("score")
            texto = (chunk.get("texto", "") or "").strip()
            search_summary = chunk.get("search_summary")
            score_components = chunk.get("score_components")
            match_terms = chunk.get("match_terms") or []
            title_match_terms = chunk.get("title_match_terms") or []

            st.markdown(
                f'<div class="rag-source-item"><span class="rag-source-index">{index}</span><strong>{html.escape(str(titulo))}</strong></div>',
                unsafe_allow_html=True,
            )
            source_caption = f"Fonte: {fonte}"
            if page_number is not None:
                source_caption += f" | p\u00e1gina: {page_number}"
            if isinstance(score, (int, float)):
                source_caption += f" | score: {float(score):.4f}"
            st.caption(source_caption)
            if page_label:
                st.caption(str(page_label))
            if search_summary:
                st.caption(str(search_summary))
            if match_terms or title_match_terms:
                details = []
                if match_terms:
                    details.append(f"termos recuperados: {', '.join(match_terms[:5])}")
                if title_match_terms:
                    details.append(f"t\u00edtulo: {', ' .join(title_match_terms[:3])}")
                st.caption(" | ".join(details))
            if isinstance(score_components, dict):
                st.caption(
                    "Componentes da busca: "
                    + ", ".join(
                        f"{name}={float(value):.4f}" for name, value in score_components.items() if isinstance(value, (int, float))
                    )
                )
            st.markdown('<div class="rag-scroll-box">', unsafe_allow_html=True)
            st.code(texto or "Trecho indisponivel.", language="text")
            st.markdown('</div>', unsafe_allow_html=True)


def _extract_pdf_page_segments(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str) or "[Pagina " not in text:
        return []

    matches = list(PDF_PAGE_MARKER_RE.finditer(text))
    segments: List[Dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        page_text = text[start:end].strip()
        if not page_text:
            continue
        page_number = int(match.group(1))
        segments.append(
            {
                "page_number": page_number,
                "page_label": f"P\u00e1gina {page_number}",
                "page_text": page_text,
            }
        )
    return segments


def _profile_pdf_document(pdf_pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    token_count = 0
    for page in pdf_pages:
        token_count += len(re.findall(r"[0-9A-Za-z_\u00c0-\u00ff]+", page.get("page_text", ""), re.UNICODE))
    short_pdf = len(pdf_pages) <= 6 and token_count <= 1500
    return {"page_count": len(pdf_pages), "token_count": token_count, "short_pdf": short_pdf}


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
            import fitz  # type: ignore

            document = fitz.open(stream=raw_bytes, filetype="pdf")
            pages = []
            for index, page in enumerate(document, start=1):
                page_text = (page.get_text("text") or "").strip()
                if page_text:
                    pages.append(f"\n\n[Pagina {index}]\n{page_text}")
            if pages:
                return "\n".join(pages)
        except Exception:
            pass

        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except Exception as exc:
                raise RuntimeError("PDF support requires pypdf, PyPDF2 or pymupdf to be installed.") from exc

        from io import BytesIO

        reader = PdfReader(BytesIO(raw_bytes))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            if page_text:
                pages.append(f"\n\n[Pagina {index}]\n{page_text}")
        return "\n".join(pages)

    raise RuntimeError("Tipo de arquivo nao suportado. Envie TXT, MD, CSV, LOG ou PDF.")


def _ingest_document(text: str, metadata: Dict[str, Any], clear_existing_chunks: bool = False) -> Dict[str, Any]:
    if not _module_ingest_and_anonymize or not _module_chunk_document or not _module_generate_embeddings or not _module_store_in_vector_db:
        raise RuntimeError("As fun\u00e7\u00f5es de ingest\u00e3o ainda n\u00e3o est\u00e3o dispon\u00edveis nos m\u00f3dulos do projeto.")

    if clear_existing_chunks and _module_clear_vector_db:
        _module_clear_vector_db()

    cleaned = _module_ingest_and_anonymize(text)
    pdf_pages = _extract_pdf_page_segments(cleaned)
    pdf_profile = _profile_pdf_document(pdf_pages) if pdf_pages else None
    chunks: List[Dict[str, Any]] = []

    if pdf_pages:
        for page in pdf_pages:
            remaining = MAX_INDEXED_CHUNKS - len(chunks)
            if remaining <= 0:
                break
            page_metadata = dict(metadata)
            page_metadata["page_number"] = page["page_number"]
            page_metadata["page_label"] = page["page_label"]
            page_metadata["page_count"] = len(pdf_pages)
            page_metadata["analysis_mode"] = "detalhado" if pdf_profile and pdf_profile["short_pdf"] else "padrao"
            page_chunks = _module_chunk_document(page["page_text"], page_metadata)
            for chunk in page_chunks:
                if len(chunks) >= MAX_INDEXED_CHUNKS:
                    break
                chunk_metadata = dict(chunk.get("metadata", {}))
                chunk_metadata["page_number"] = page["page_number"]
                chunk_metadata["page_label"] = page["page_label"]
                chunk_metadata["page_count"] = len(pdf_pages)
                chunk_metadata["analysis_mode"] = "detalhado" if pdf_profile and pdf_profile["short_pdf"] else "padrao"
                chunk["metadata"] = chunk_metadata
                chunks.append(chunk)
    else:
        chunks = _module_chunk_document(cleaned, metadata)[:MAX_INDEXED_CHUNKS]

    embedded = _module_generate_embeddings(chunks)
    stored = bool(_module_store_in_vector_db(embedded))

    st.session_state.last_ingestion = {
        "metadata": metadata,
        "cleaned_text": cleaned,
        "chunks_count": len(chunks),
        "stored": stored,
        "doc_id": chunks[0]["doc_id"] if chunks else None,
        "page_count": len(pdf_pages) if pdf_pages else None,
        "pdf_profile": pdf_profile,
    }

    return {"cleaned_text": cleaned, "chunks": embedded, "stored": stored}


def _retrieve_context(question: str, top_k: int = 15) -> List[Dict[str, Any]]:
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
    if _module_search_similar_chunks:
        try:
            return _module_search_similar_chunks(question, top_k=top_k)
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
        return {"resposta_gerada": str(result), "fontes": context, "precisou_triagem": False, "confianca": None}

    return {
        "resposta_gerada": "Nenhum m\u00f3dulo de busca ou resposta est\u00e1 dispon\u00edvel no momento.",
        "fontes": context,
        "precisou_triagem": True,
        "confianca": 0.0,
    }


def _submit_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question, "sources": None, "interaction_id": None})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando contexto e gerando resposta..."):
            context = _retrieve_context(question, top_k=20)
            result = _generate_answer(question, context)

        _render_scrollable_text(result["resposta_gerada"])
        if result.get("resumo_busca"):
            st.markdown('<div class="rag-mini-card">', unsafe_allow_html=True)
            st.markdown("**Resumo da pesquisa**")
            st.write(result["resumo_busca"])
            st.markdown("</div>", unsafe_allow_html=True)
        if result.get("resumo_documento"):
            st.markdown('<div class="rag-mini-card">', unsafe_allow_html=True)
            st.markdown("**Sintese do documento**")
            st.write(result["resumo_documento"])
            st.markdown("</div>", unsafe_allow_html=True)
        if result.get("analise_documento"):
            st.markdown('<div class="rag-mini-card">', unsafe_allow_html=True)
            st.markdown("**Analise do conteudo**")
            st.write(result["analise_documento"])
            st.markdown("</div>", unsafe_allow_html=True)
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
        {"role": "assistant", "content": result["resposta_gerada"], "sources": result.get("fontes", []), "interaction_id": interaction["id"]}
    )


def _render_chat_history() -> None:
    for message in st.session_state.messages:
        role = message.get("role", "assistant")
        content = message.get("content", "")
        with st.chat_message(role):
            st.markdown(content)


def render_chat_interface() -> None:
    st.set_page_config(page_title="Chatbot Ragnarok", page_icon="R", layout="wide")
    _ensure_state()
    _inject_styles()

    left_col, right_col = st.columns([1.75, 1], gap="large")

    with st.sidebar:
        st.header("Base de conhecimento")
        st.markdown("Carregue um documento ou cole um texto para ingestao.")

        uploaded_file = st.file_uploader("Documento", type=["txt", "md", "csv", "log", "pdf"], accept_multiple_files=False)
        document_title = st.text_input("Titulo do documento", value="Documento carregado")
        document_source = st.text_input("Fonte original", value="documento")
        clear_existing_chunks = st.checkbox("Limpar trechos anteriores ao inserir", value=True)

        sample_text = st.text_area(
            "Ou cole um texto para ingestao rapida",
            height=170,
            placeholder="Cole aqui o conteudo bruto do documento...",
        )

        if st.button("Ingerir documento", use_container_width=True):
            try:
                raw_text = sample_text.strip()
                source_label = document_source.strip() or "documento"
                original_name = uploaded_file.name if uploaded_file is not None else source_label
                if uploaded_file is not None:
                    raw_text = _extract_text_from_upload(uploaded_file)

                if not raw_text:
                    st.warning("Forneca um arquivo ou cole um texto para ingestao.")
                else:
                    metadata = {
                        "titulo": document_title.strip() or "Documento sem titulo",
                        "fonte": source_label,
                        "arquivo_origem": original_name,
                        "tipo": Path(original_name).suffix.lstrip(".") or "texto",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                    result = _ingest_document(raw_text, metadata, clear_existing_chunks=clear_existing_chunks)
                    st.success(f"Documento ingerido com sucesso. {len(result['chunks'])} trechos indexados.")
            except Exception as exc:
                st.error(f"Falha na ingestao: {exc}")

        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.interactions = []
            st.rerun()

        st.divider()
        if _module_get_collection:
            try:
                indexed_count = _module_get_collection().count()
            except Exception:
                indexed_count = 0
        elif _module_load_vector_db:
            indexed_count = len(_module_load_vector_db())
        else:
            indexed_count = 0
        st.metric("Chunks indexados", indexed_count)
        if st.session_state.last_ingestion:
            st.caption(f"Ultima ingestao: {st.session_state.last_ingestion['doc_id']}")
            if st.session_state.last_ingestion.get("page_count"):
                st.caption(f"Paginas detectadas: {st.session_state.last_ingestion['page_count']}")
            if st.session_state.last_ingestion.get("pdf_profile"):
                profile = st.session_state.last_ingestion["pdf_profile"]
                if profile.get("short_pdf"):
                    st.caption("PDF curto detectado: an\u00e1lise detalhada ativada.")
                else:
                    st.caption("PDF longo detectado: an\u00e1lise equilibrada ativada.")
        else:
            st.markdown('<div class="rag-panel rag-muted">As respostas aparecer\u00e3o aqui depois da primeira pergunta.</div>', unsafe_allow_html=True)

    with left_col:
        st.markdown(
            """<div class="rag-panel">
                <div class="rag-title">Chatbot Ragnarok</div>
                <div class="rag-subtitle">An\u00e1lise de documentos com mini-RAG, resposta fundamentada e foco em PT-BR padr\u00e3o.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        _render_chat_history()
        prompt = st.chat_input("Pergunte sobre o documento carregado")
        if prompt:
            _submit_question(prompt)

    with right_col:
        st.subheader("Status")
        st.markdown(
            '<div class="rag-muted">Use a \u00e1rea principal para perguntar sobre os documentos carregados. Se a evid\u00eancia for fraca, o app marca triagem humana.</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.last_ingestion:
            st.markdown('<div class="rag-status">Documento ingerido e pronto para busca.</div>', unsafe_allow_html=True)
            st.json(
                {
                    "doc_id": st.session_state.last_ingestion["doc_id"],
                    "chunks": st.session_state.last_ingestion["chunks_count"],
                    "stored": st.session_state.last_ingestion["stored"],
                }
            )
        else:
            st.info("Nenhum documento foi ingerido ainda.")


if __name__ == "__main__":
    render_chat_interface()
