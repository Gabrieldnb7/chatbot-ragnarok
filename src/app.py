"""Interface Streamlit do chatbot RAGnarok."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma_data"
COLLECTION_NAME = "ragnarok_knowledge_base"
DEFAULT_TOP_K = 5


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .stApp { background: #000000; color: #f3f4f6; }
        .main .block-container {
            max-width: 100%;
            padding: 1.3rem 2rem 7rem;
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background: transparent;
        }
        .stButton button,
        .stForm button { border-radius: 8px; }
        .rag-chat-wrap {
            max-width: 760px;
            height: calc(100vh - 2.6rem);
            margin: 0 auto;
            padding: 0 0 1.5rem;
            display: flex;
            flex-direction: column;
        }
        .rag-chat-header {
            position: sticky;
            top: 0;
            z-index: 50;
            text-align: center;
            padding: 1.15rem 0 0.85rem;
            background: linear-gradient(180deg, #000 72%, rgba(0, 0, 0, 0));
        }
        .rag-chat-scroll {
            flex: 1;
            overflow-y: auto;
            padding: 0 0.25rem 7rem;
            scroll-behavior: smooth;
            overscroll-behavior: contain;
            scrollbar-width: thin;
        }
        .rag-chat-bottom {
            height: 1px;
            width: 100%;
        }
        .rag-title {
            font-size: 1.55rem;
            font-weight: 750;
            line-height: 1.15;
            margin-bottom: 0.35rem;
        }
        .rag-subtitle {
            color: rgba(243, 244, 246, 0.62);
            font-size: 0.92rem;
            margin: 0 auto;
            max-width: 560px;
        }
        .rag-empty-state {
            color: rgba(243, 244, 246, 0.86);
            font-size: 1.35rem;
            font-weight: 520;
            text-align: center;
            padding: 32vh 1rem 0;
        }
        .rag-side-panel {
            min-height: auto;
            border-radius: 12px;
            padding: 1rem 0.95rem;
            background: rgba(8, 9, 12, 0.72);
            border: 1px solid rgba(255, 255, 255, 0.07);
            position: sticky;
            top: 1rem;
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
        .rag-muted { color: rgba(243, 244, 246, 0.72); font-size: 0.92rem; }
        .rag-message-row {
            display: flex;
            margin: 1.05rem 0;
            width: 100%;
        }
        .rag-message-row.user { justify-content: flex-end; }
        .rag-message-row.assistant { justify-content: flex-start; }
        .rag-message {
            max-width: 86%;
            line-height: 1.68;
            font-size: 1rem;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .rag-message.user {
            background: #2f3037;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 0.72rem 0.95rem;
            color: #f3f4f6;
        }
        .rag-message.assistant {
            color: #f3f4f6;
            padding: 0.2rem 0;
        }
        .rag-mini-card,
        .rag-rank-card {
            background: #0f131b;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-top: 0.8rem;
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
            max-height: 260px;
            overflow-y: auto;
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
                min-height: auto;
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
    llm_config = st.session_state.get("llm_config")
    return generate_rag_response(question, context, llm_config)


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


def _clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.last_top_k = []
    st.session_state.last_question = ""
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

    with st.expander("Configurações do LLM", expanded=False):
        provider = st.selectbox(
            "Provedor",
            ["deepseek", "gemini", "local"],
            index=0,
            key="llm_provider",
        )
        model = st.text_input(
            "Modelo",
            value=st.session_state.get("llm_model", "deepseek-chat"),
            key="llm_model",
        )
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.get("llm_api_key", ""),
            key="llm_api_key",
        )

        if provider == "local":
            st.caption("Modo local: respostas analíticas sem API. Nenhuma chave necessária.")
            st.session_state.pop("llm_config", None)
        elif api_key:
            st.session_state.llm_config = {
                "provider": provider,
                "model": model,
                "api_key": api_key,
            }
        else:
            st.session_state.pop("llm_config", None)
            st.caption(f"Defina a API Key ou configure a env {provider.upper()}_API_KEY")
    st.markdown("</div>", unsafe_allow_html=True)


def _score_label(chunk: dict[str, Any]) -> str:
    score = chunk.get("score")
    if isinstance(score, (int, float)):
        return f"{float(score):.4f}"
    return "--"


def _render_rank_card(index: int, chunk: dict[str, Any]) -> None:
    metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
    title = metadata.get("titulo") or "Documento"
    source = metadata.get("fonte") or "documento"
    text = (chunk.get("texto", "") if isinstance(chunk, dict) else "").strip()
    preview = text[:520] + ("..." if len(text) > 520 else "")

    st.markdown(
        f"""
        <div class="rag-rank-card">
            <div class="rag-rank-head">
                <span class="rag-rank-number">#{index}</span>
                <span class="rag-score">score {_score_label(chunk)}</span>
            </div>
            <div class="rag-rank-title">{html.escape(str(title))}</div>
            <div class="rag-rank-source">Origem: {html.escape(str(source))}</div>
            <div class="rag-rank-preview">{html.escape(preview)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_k_panel() -> None:
    st.markdown('<div class="rag-side-panel">', unsafe_allow_html=True)
    st.markdown('<div class="rag-side-title">Top-k encontrados</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rag-side-kicker">Ranking dos chunks retornados pela última pergunta, do melhor score para baixo.</div>',
        unsafe_allow_html=True,
    )

    top_k = st.session_state.get("last_top_k") or []
    if not top_k:
        st.info("Faça uma pergunta para ver os chunks mais relevantes da base.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if st.session_state.get("last_question"):
        st.caption(f"Pergunta: {st.session_state.last_question}")

    ranked_chunks = sorted(
        top_k,
        key=lambda chunk: float(chunk.get("score", 0.0)) if isinstance(chunk, dict) else 0.0,
        reverse=True,
    )
    for index, chunk in enumerate(ranked_chunks, start=1):
        _render_rank_card(index, chunk)
    st.markdown("</div>", unsafe_allow_html=True)


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