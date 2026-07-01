"""Interface Streamlit do chatbot RAGnarok."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma_data"
COLLECTION_NAME = "ragnarok_knowledge_base"
DEFAULT_TOP_K = 10


def _inject_styles() -> None:
    st.markdown(
        """
                <style>
        :root { color-scheme: dark; }
        .stApp {
            background: #0b141a;
            color: #e9edef;
        }
        .main .block-container {
            max-width: 100%;
            padding: 1rem 1.25rem 6.75rem;
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: transparent;
        }
        .stButton button,
        .stForm button {
            border-radius: 8px;
        }
        .rag-chat-wrap {
            max-width: 820px;
            height: calc(100vh - 2.2rem);
            min-height: 0;
            margin: 0 auto;
            padding: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background: #0b141a;
        }
        .rag-chat-header {
            flex: 0 0 auto;
            z-index: 50;
            padding: 0.95rem 0.95rem 0.8rem;
            background: #202c33;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 8px 8px 0 0;
        }
        .rag-chat-scroll {
            flex: 1 1 auto;
            min-height: 0;
            height: 100%;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 1rem 0.95rem 7.25rem;
            scroll-behavior: smooth;
            overscroll-behavior: contain;
            scrollbar-gutter: stable;
            scrollbar-width: thin;
            scrollbar-color: rgba(134, 150, 160, 0.55) rgba(255, 255, 255, 0.04);
            background:
                linear-gradient(rgba(11, 20, 26, 0.93), rgba(11, 20, 26, 0.93)),
                repeating-linear-gradient(135deg, rgba(255,255,255,0.03) 0 1px, transparent 1px 18px);
        }
        .rag-chat-scroll::-webkit-scrollbar {
            width: 10px;
        }
        .rag-chat-scroll::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.04);
            border-radius: 999px;
        }
        .rag-chat-scroll::-webkit-scrollbar-thumb {
            background: rgba(134, 150, 160, 0.55);
            border-radius: 999px;
            border: 2px solid #0b141a;
        }
        .rag-chat-bottom {
            height: 1px;
            width: 100%;
        }
        .rag-title {
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.18rem;
            color: #e9edef;
        }
        .rag-subtitle {
            color: #8696a0;
            font-size: 0.82rem;
            line-height: 1.35;
        }
        .rag-empty-state {
            color: #d1d7db;
            font-size: 1.2rem;
            font-weight: 520;
            text-align: center;
            padding: 24vh 1rem 0;
        }
        .rag-side-panel {
            border-radius: 8px;
            padding: 1rem 0.95rem;
            background: rgba(8, 9, 12, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.07);
            position: sticky;
            top: 1rem;
        }
        .rag-topk-panel {
            height: calc(100vh - 2.2rem);
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            padding: 0;
            background: #05090c;
            border-color: rgba(255, 255, 255, 0.08);
        }
        .rag-topk-header {
            flex: 0 0 auto;
            padding: 0.95rem 0.95rem 0.75rem;
            background: #0b141a;
            border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        }
        .rag-topk-scroll {
            flex: 1 1 auto;
            min-height: 0;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 0.75rem 0.95rem 0.95rem;
            overscroll-behavior: contain;
            scrollbar-gutter: stable;
            scrollbar-width: thin;
            scrollbar-color: rgba(134, 150, 160, 0.55) rgba(255, 255, 255, 0.04);
        }
        .rag-topk-scroll::-webkit-scrollbar {
            width: 10px;
        }
        .rag-topk-scroll::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.04);
            border-radius: 999px;
        }
        .rag-topk-scroll::-webkit-scrollbar-thumb {
            background: rgba(134, 150, 160, 0.55);
            border-radius: 999px;
            border: 2px solid #0b141a;
        }
        .rag-side-title {
            font-size: 1rem;
            font-weight: 750;
            margin-bottom: 0.55rem;
        }
        .rag-side-kicker {
            color: rgba(243, 244, 246, 0.58);
            font-size: 0.82rem;
            line-height: 1.4;
            margin-bottom: 1rem;
        }
        .rag-topk-question {
            color: #cfd6dc;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 0.72rem;
            overflow-wrap: anywhere;
        }
        .rag-topk-empty {
            color: #8696a0;
            font-size: 0.88rem;
            line-height: 1.45;
            padding: 0.85rem 0.1rem;
        }
        .rag-message-row {
            display: flex;
            margin: 0.28rem 0;
            width: 100%;
        }
        .rag-message-row.user { justify-content: flex-end; }
        .rag-message-row.assistant { justify-content: flex-start; }
        .rag-message {
            position: relative;
            max-width: min(78%, 620px);
            line-height: 1.45;
            font-size: 0.96rem;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            padding: 0.52rem 0.72rem 1.22rem;
            box-shadow: 0 1px 0.5px rgba(0, 0, 0, 0.22);
        }
        .rag-message.user {
            background: #005c4b;
            border-radius: 8px 0 8px 8px;
            color: #e9edef;
        }
        .rag-message.user::after {
            content: "";
            position: absolute;
            right: -7px;
            top: 0;
            border-top: 8px solid #005c4b;
            border-right: 8px solid transparent;
        }
        .rag-message.assistant {
            background: #202c33;
            color: #e9edef;
            border-radius: 0 8px 8px 8px;
        }
        .rag-message.assistant::before {
            content: "";
            position: absolute;
            left: -7px;
            top: 0;
            border-top: 8px solid #202c33;
            border-left: 8px solid transparent;
        }
        .rag-message-meta {
            position: absolute;
            right: 0.58rem;
            bottom: 0.32rem;
            color: rgba(233, 237, 239, 0.56);
            font-size: 0.69rem;
            line-height: 1;
        }
        .rag-message.user .rag-message-meta::after {
            content: "  \2713\2713";
            color: #53bdeb;
            letter-spacing: -0.08rem;
        }
        .rag-details {
            margin-top: 0.56rem;
            border-top: 1px solid rgba(255, 255, 255, 0.08);
            padding-top: 0.5rem;
        }
        .rag-details summary {
            cursor: pointer;
            color: #53bdeb;
            font-size: 0.84rem;
            list-style: none;
        }
        .rag-details summary::-webkit-details-marker { display: none; }
        .rag-mini-card,
        .rag-rank-card {
            background: rgba(17, 27, 33, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.07);
            border-radius: 8px;
            padding: 0.72rem 0.82rem;
            margin-top: 0.62rem;
        }
        .rag-mini-title {
            color: #f3f4f6;
            font-weight: 750;
            margin-bottom: 0.35rem;
        }
        .rag-mini-body {
            color: rgba(243, 244, 246, 0.78);
            line-height: 1.55;
        }
        .rag-rank-card {
            overflow: hidden;
        }
        .rag-rank-head {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.45rem;
        }
        .rag-rank-number {
            color: #d7eafe;
            font-weight: 800;
            font-size: 1rem;
        }
        .rag-score {
            background: rgba(15, 58, 93, 0.9);
            color: #d7eafe;
            border-radius: 999px;
            padding: 0.18rem 0.55rem;
            font-size: 0.78rem;
            font-weight: 700;
            white-space: nowrap;
        }
        .rag-rank-title {
            color: #f3f4f6;
            font-weight: 700;
            line-height: 1.25;
            margin-bottom: 0.25rem;
        }
        .rag-rank-source,
        .rag-rank-preview {
            color: rgba(243, 244, 246, 0.72);
            font-size: 0.88rem;
            line-height: 1.45;
        }
        [data-testid="stChatInput"] {
            position: fixed;
            left: 50%;
            right: auto;
            bottom: 1.35rem;
            width: min(760px, calc(100vw - 33rem));
            transform: translateX(-50%);
            z-index: 100;
        }
        [data-testid="stChatInput"] > div {
            border-radius: 22px;
            background: #202123;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.34);
        }
        @media (max-width: 1100px) {
            [data-testid="stChatInput"] {
                width: min(760px, calc(100vw - 2rem));
            }
            .rag-side-panel {
                position: static;
            }
        }
        @media (max-width: 900px) {
            .main .block-container { padding: 1rem 1rem 7rem; }
            [data-testid="stChatInput"] {
                left: 1rem;
                right: 1rem;
                width: auto;
                transform: none;
                bottom: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_top_k", [])
    st.session_state.setdefault("last_question", "")
    st.session_state.setdefault("pending_prompt", None)


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


def get_indexed_count() -> int:
    try:
        return int(get_collection().count())
    except Exception:
        return 0


def generate_answer(question: str) -> dict[str, Any]:
    """Recupera contexto, registra o top-k e gera a resposta RAG."""
    from llm_integration import generate_rag_response
    from retrieval import retrieve_context

    context = retrieve_context(question, top_k=DEFAULT_TOP_K)
    st.session_state.last_top_k = context
    st.session_state.last_question = question
    return generate_rag_response(question, context)


def _clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.last_top_k = []
    st.session_state.last_question = ""
    st.session_state.pending_prompt = None
    st.rerun()


def render_metrics_panel() -> None:
    st.markdown('<div class="rag-side-panel">', unsafe_allow_html=True)
    st.markdown('<div class="rag-side-title">Métricas</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rag-side-kicker">Indicadores da base vetorial do projeto.</div>',
        unsafe_allow_html=True,
    )
    st.metric("Chunks gerados", get_indexed_count())
    st.caption(f"Coleção: {COLLECTION_NAME}")

    if st.button("Limpar conversa", use_container_width=True):
        _clear_conversation()
    st.markdown("</div>", unsafe_allow_html=True)



def _score_label(chunk: dict[str, Any]) -> str:
    score = chunk.get("score")
    if isinstance(score, (int, float)):
        return f"{float(score):.4f}"
    return "--"


def _topk_card_html(index: int, chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
    title = metadata.get("titulo") or "Documento"
    source = metadata.get("fonte") or "documento"
    text = (chunk.get("texto", "") if isinstance(chunk, dict) else "").strip()
    preview = text[:520] + ("..." if len(text) > 520 else "")

    return f"""
        <div class="rag-rank-card">
            <div class="rag-rank-head">
                <span class="rag-rank-number">#{index}</span>
                <span class="rag-score">score {_score_label(chunk)}</span>
            </div>
            <div class="rag-rank-title">{html.escape(str(title))}</div>
            <div class="rag-rank-source">Origem: {html.escape(str(source))}</div>
            <div class="rag-rank-preview">{html.escape(preview)}</div>
        </div>
    """


def render_top_k_panel() -> None:
    top_k = st.session_state.get("last_top_k") or []
    if not top_k:
        st.html(
            """
            <div class="rag-side-panel rag-topk-panel">
                <div class="rag-topk-header">
                    <div class="rag-side-title">Top-k encontrados</div>
                    <div class="rag-side-kicker">Ranking dos chunks retornados pela última pergunta, do melhor score para baixo.</div>
                </div>
                <div class="rag-topk-scroll">
                    <div class="rag-topk-empty">Faça uma pergunta para ver os chunks mais relevantes da base.</div>
                </div>
            </div>
            """
        )
        return

    question_html = ""
    if st.session_state.get("last_question"):
        question_html = f'<div class="rag-topk-question">Pergunta: {html.escape(str(st.session_state.last_question))}</div>'

    ranked_chunks = sorted(
        top_k,
        key=lambda chunk: float(chunk.get("score", 0.0)) if isinstance(chunk, dict) else 0.0,
        reverse=True,
    )
    cards_html = "".join(_topk_card_html(index, chunk) for index, chunk in enumerate(ranked_chunks, start=1))

    st.html(
        f"""
        <div class="rag-side-panel rag-topk-panel">
            <div class="rag-topk-header">
                <div class="rag-side-title">Top-k encontrados</div>
                <div class="rag-side-kicker">Ranking dos chunks retornados pela última pergunta, do melhor score para baixo.</div>
                {question_html}
            </div>
            <div class="rag-topk-scroll">
                {cards_html}
            </div>
        </div>
        """
    )
def _text_to_html(value: Any) -> str:
    return html.escape(str(value or "")).replace("\n", "<br>")


def _response_details_html(response: dict[str, Any] | None) -> str:
    if not response:
        return ""

    cards = []
    detail_labels = (
        ("resumo_busca", "Resumo da pesquisa"),
        ("resumo_documento", "Sintese do documento"),
        ("analise_documento", "Analise do conteudo"),
    )
    for key, label in detail_labels:
        value = response.get(key)
        if value:
            cards.append(
                '<div class="rag-mini-card">'
                f'<div class="rag-mini-title">{html.escape(label)}</div>'
                f'<div class="rag-mini-body">{_text_to_html(value)}</div>'
                '</div>'
            )
    return "".join(cards)


def _chat_message_html(role: str, content: str, response: dict[str, Any] | None = None) -> str:
    normalized_role = "user" if role == "user" else "assistant"
    return (
        f'<div class="rag-message-row {normalized_role}">'
        f'<div class="rag-message {normalized_role}">{_text_to_html(content)}</div>'
        '</div>'
        + _response_details_html(response if normalized_role == "assistant" else None)
    )


def render_chat_history() -> None:
    if st.session_state.messages:
        body = "".join(
            _chat_message_html(
                message.get("role", "assistant"),
                message.get("content", ""),
                message.get("response"),
            )
            for message in st.session_state.messages
        )
    else:
        body = '<div class="rag-empty-state">O que voce quer saber sobre a base?</div>'

    st.markdown(
        f"""
        <div class="rag-chat-wrap">
            <div class="rag-chat-header">
                <div class="rag-title">Chatbot Ragnarok</div>
                <div class="rag-subtitle">Analise de documentos com mini-RAG, resposta fundamentada e foco em PT-BR padrao.</div>
            </div>
            <div class="rag-chat-scroll">
                {body}
                <div class="rag-chat-bottom"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _auto_scroll_chat() -> None:
    st.html(
        """
        <script>
        const scrollChatToBottom = () => {
            const parentDoc = window.parent.document;
            const scrollArea = parentDoc.querySelector(".rag-chat-scroll");
            if (!scrollArea) return;

            scrollArea.scrollTo({
                top: scrollArea.scrollHeight,
                behavior: "smooth"
            });
        };

        const setupAutoScroll = () => {
            const parentDoc = window.parent.document;
            const scrollArea = parentDoc.querySelector(".rag-chat-scroll");
            if (!scrollArea) return;

            if (!scrollArea.dataset.autoScrollReady) {
                const observer = new MutationObserver(() => {
                    window.requestAnimationFrame(scrollChatToBottom);
                });
                observer.observe(scrollArea, { childList: true, subtree: true });
                scrollArea.dataset.autoScrollReady = "true";
            }

            scrollChatToBottom();
        };

        window.setTimeout(setupAutoScroll, 50);
        window.setTimeout(scrollChatToBottom, 250);
        window.setTimeout(scrollChatToBottom, 700);
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def render_chat_interface() -> None:
    st.set_page_config(page_title="Chatbot Ragnarok", page_icon="R", layout="wide")
    _ensure_state()
    _inject_styles()

    metrics_col, chat_col, top_k_col = st.columns([0.72, 2.25, 0.9], gap="large")

    with metrics_col:
        render_metrics_panel()

    with chat_col:
        render_chat_history()
        _auto_scroll_chat()

        pending_prompt = st.session_state.get("pending_prompt")
        if pending_prompt:
            with st.spinner("Buscando contexto e gerando resposta..."):
                try:
                    response = generate_answer(pending_prompt)
                except Exception as exc:
                    st.session_state.last_top_k = []
                    st.session_state.last_question = pending_prompt
                    response = {
                        "resposta_gerada": f"Nao foi possivel responder agora: {exc}",
                        "fontes": [],
                        "precisou_triagem": True,
                        "confianca": 0.0,
                    }
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response["resposta_gerada"],
                    "response": response,
                }
            )
            st.session_state.pending_prompt = None
            st.rerun()

        prompt = st.chat_input("Pergunte sobre os documentos da base")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.pending_prompt = prompt
            st.rerun()

    with top_k_col:
        render_top_k_panel()


if __name__ == "__main__":
    render_chat_interface()
