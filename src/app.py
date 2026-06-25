"""Interface Streamlit do chatbot RAGnarok."""

from __future__ import annotations

import html
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma_data"
COLLECTION_NAME = "ragnarok_knowledge_base"
DEFAULT_TOP_K = 5
SUPPORTED_UPLOAD_TYPES = ["txt", "md", "csv", "log", "pdf"]


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #0b0d12; color: #f3f4f6; }
        [data-testid="stSidebar"] {
            background: #1b1d26;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }
        [data-testid="stSidebar"] .stButton button,
        .stButton button,
        .stForm button { border-radius: 8px; }
        .rag-panel {
            background: #11131a;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1rem 1.1rem;
        }
        .rag-title {
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.35rem;
        }
        .rag-subtitle {
            color: rgba(243, 244, 246, 0.7);
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .rag-muted { color: rgba(243, 244, 246, 0.72); font-size: 0.92rem; }
        .rag-status {
            background: #0f3a5d;
            color: #d7eafe;
            border-radius: 10px;
            padding: 0.9rem 1rem;
            margin-top: 0.8rem;
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
    st.session_state.setdefault("last_ingestion", None)


@st.cache_resource(show_spinner=False)
def get_collection():
    """Retorna a coleção Chroma usada pelo app."""
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(
        path=str(VECTOR_DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def decode_text_file(raw_content: bytes) -> str:
    """Decodifica arquivos textuais enviados pela interface."""
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw_content.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw_content.decode("utf-8", errors="replace")


def extract_pdf_text(raw_content: bytes) -> str:
    """Extrai texto de PDF usando bibliotecas opcionais, se estiverem instaladas."""
    try:
        import fitz  # type: ignore
    except Exception:
        fitz = None

    if fitz is not None:
        try:
            document = fitz.open(stream=raw_content, filetype="pdf")
            pages = []
            for index, page in enumerate(document, start=1):
                page_text = (page.get_text("text") or "").strip()
                if page_text:
                    pages.append(f"\n\n[Página {index}]\n{page_text}")
            if pages:
                return "\n".join(pages)
        except Exception:
            pass

    PdfReader = None
    try:
        from pypdf import PdfReader as PypdfReader  # type: ignore

        PdfReader = PypdfReader
    except Exception:
        try:
            from PyPDF2 import PdfReader as PyPdf2Reader  # type: ignore

            PdfReader = PyPdf2Reader
        except Exception as exc:
            raise RuntimeError(
                "Para importar PDF, instale uma biblioteca de leitura de PDF: pymupdf, pypdf ou PyPDF2."
            ) from exc

    pages = []
    reader = PdfReader(BytesIO(raw_content))
    for index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if page_text:
            pages.append(f"\n\n[Página {index}]\n{page_text}")

    return "\n".join(pages)


def _extract_text_from_upload(uploaded_file) -> str:
    """Extrai texto dos mesmos formatos aceitos pelo layout de referencia."""
    if uploaded_file is None:
        return ""

    raw_content = uploaded_file.getvalue()
    file_suffix = Path(uploaded_file.name or "").suffix.lower()

    if file_suffix == ".pdf":
        text = extract_pdf_text(raw_content)
    elif file_suffix in {".txt", ".md", ".csv", ".log"}:
        text = decode_text_file(raw_content)
    else:
        supported = ", ".join(SUPPORTED_UPLOAD_TYPES)
        raise RuntimeError(f"Tipo de arquivo não suportado. Use: {supported}.")

    if not text.strip():
        raise RuntimeError("Não foi possível extrair texto do arquivo enviado.")

    return text


def index_document(raw_text: str, title: str, source: str) -> int:
    """Executa ingestão, chunking, embedding e persistência no Chroma."""
    from chunking import chunk_document
    from embeddings import generate_embeddings
    from ingestion import ingest_and_anonymize

    cleaned_text = ingest_and_anonymize(raw_text)
    metadata = {
        "titulo": title.strip() or "Documento sem título",
        "fonte": source.strip() or "Entrada manual",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    chunks = chunk_document(cleaned_text, metadata)
    embedded_chunks = generate_embeddings(chunks)

    if not embedded_chunks:
        return 0

    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in embedded_chunks],
        embeddings=[chunk["embedding"] for chunk in embedded_chunks],
        documents=[chunk["texto"] for chunk in embedded_chunks],
        metadatas=[chunk.get("metadata", {}) for chunk in embedded_chunks],
    )

    st.session_state.last_ingestion = {
        "titulo": metadata["titulo"],
        "fonte": metadata["fonte"],
        "chunks_count": len(embedded_chunks),
    }
    return len(embedded_chunks)


def generate_answer(question: str) -> dict[str, Any]:
    """Recupera contexto e gera a resposta RAG."""
    from llm_integration import generate_rag_response
    from retrieval import retrieve_context

    context = retrieve_context(question, top_k=DEFAULT_TOP_K)
    return generate_rag_response(question, context)


def _render_scrollable_text(text: str) -> None:
    safe_text = html.escape(text or "")
    st.markdown(f'<div class="rag-answer-box">{safe_text}</div>', unsafe_allow_html=True)


def _render_result_details(response: dict[str, Any]) -> None:
    if response.get("resumo_busca"):
        st.markdown('<div class="rag-mini-card">', unsafe_allow_html=True)
        st.markdown("**Resumo da pesquisa**")
        st.write(response["resumo_busca"])
        st.markdown("</div>", unsafe_allow_html=True)

    if response.get("resumo_documento"):
        st.markdown('<div class="rag-mini-card">', unsafe_allow_html=True)
        st.markdown("**Síntese do documento**")
        st.write(response["resumo_documento"])
        st.markdown("</div>", unsafe_allow_html=True)

    if response.get("analise_documento"):
        st.markdown('<div class="rag-mini-card">', unsafe_allow_html=True)
        st.markdown("**Análise do conteúdo**")
        st.write(response["analise_documento"])
        st.markdown("</div>", unsafe_allow_html=True)

    sources = response.get("fontes") or []
    if sources:
        with st.expander(f"Fontes consultadas ({len(sources)})"):
            for index, chunk in enumerate(sources, start=1):
                metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                title = metadata.get("titulo") or "Documento"
                source = metadata.get("fonte") or "documento"
                score = chunk.get("score") if isinstance(chunk, dict) else None
                text = (chunk.get("texto", "") if isinstance(chunk, dict) else "").strip()
                score_line = f"\nScore: {float(score):.4f}" if isinstance(score, (int, float)) else ""
                source_text = f"Fonte {index}: {title}\nOrigem: {source}{score_line}\n\nTrecho:\n{text}"
                _render_scrollable_text(source_text)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Base de conhecimento")
        st.markdown("Carregue um documento ou cole um texto para ingestão.")

        uploaded_file = st.file_uploader(
            "Documento",
            type=SUPPORTED_UPLOAD_TYPES,
            accept_multiple_files=False,
        )
        document_title = st.text_input("Título do documento", value="Documento carregado")
        document_source = st.text_input("Fonte original", value="documento")
        sample_text = st.text_area(
            "Ou cole um texto para ingestão rápida",
            height=170,
            placeholder="Cole aqui o conteúdo bruto do documento...",
        )

        if st.button("Ingerir documento", use_container_width=True):
            try:
                raw_text = ""
                source = document_source.strip() or "documento"

                if uploaded_file is not None:
                    raw_text = _extract_text_from_upload(uploaded_file).strip()
                    if not document_source.strip():
                        source = uploaded_file.name or source
                else:
                    raw_text = sample_text.strip()

                if not raw_text:
                    st.warning("Forneça um arquivo ou cole um texto para ingestão.")
                else:
                    with st.spinner("Processando e indexando documento..."):
                        chunks_count = index_document(raw_text, document_title, source)
                    st.success(f"Documento ingerido com sucesso. {chunks_count} trechos indexados.")
            except Exception as exc:
                st.error(f"Falha na ingestão: {exc}")

        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        try:
            indexed_count = get_collection().count()
        except Exception:
            indexed_count = 0
        st.metric("Chunks indexados", indexed_count)

        if st.session_state.last_ingestion:
            st.caption(f"Última ingestão: {st.session_state.last_ingestion['titulo']}")
            st.caption(f"Trechos: {st.session_state.last_ingestion['chunks_count']}")
        else:
            st.markdown(
                '<div class="rag-panel rag-muted">Nenhum documento ingerido ainda.</div>',
                unsafe_allow_html=True,
            )


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message.get("role", "assistant")):
            if message.get("role") == "assistant":
                _render_scrollable_text(message.get("content", ""))
                if message.get("response"):
                    _render_result_details(message["response"])
            else:
                st.markdown(message.get("content", ""))


def render_chat_interface() -> None:
    st.set_page_config(page_title="Chatbot Ragnarok", page_icon="R", layout="wide")
    _ensure_state()
    _inject_styles()

    left_col, right_col = st.columns([1.75, 1], gap="large")
    render_sidebar()

    with left_col:
        st.markdown(
            """<div class="rag-panel">
                <div class="rag-title">Chatbot Ragnarok</div>
                <div class="rag-subtitle">Análise de documentos com mini-RAG, resposta fundamentada e foco em PT-BR padrão.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        render_chat_history()

        prompt = st.chat_input("Pergunte sobre o documento carregado")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Buscando contexto e gerando resposta..."):
                    try:
                        response = generate_answer(prompt)
                    except Exception as exc:
                        response = {
                            "resposta_gerada": f"Não foi possível responder agora: {exc}",
                            "fontes": [],
                            "precisou_triagem": True,
                            "confianca": 0.0,
                        }
                _render_scrollable_text(response["resposta_gerada"])
                _render_result_details(response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["resposta_gerada"],
                    "response": response,
                }
            )

    with right_col:
        st.subheader("Status")
        st.markdown(
            '<div class="rag-muted">Use a área principal para perguntar sobre os documentos carregados. Se a evidência for fraca, o app marca triagem humana.</div>',
            unsafe_allow_html=True,
        )
        try:
            indexed_count = get_collection().count()
        except Exception:
            indexed_count = 0

        if indexed_count:
            st.markdown('<div class="rag-status">Base pronta para busca.</div>', unsafe_allow_html=True)
            st.json({"collection": COLLECTION_NAME, "chunks": indexed_count})
        else:
            st.info("Nenhum documento foi ingerido ainda.")


if __name__ == "__main__":
    render_chat_interface()