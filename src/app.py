"""Interface Streamlit do chatbot RAGnarok."""

from __future__ import annotations

import importlib
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _optional_import(module_names: List[str], attr: str):
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        candidate = getattr(module, attr, None)
        if callable(candidate):
            return candidate
    return None


retrieve_context = _optional_import(["retrieval", "src.retrieval"], "retrieve_context")
retrieve_relevant_chunks = _optional_import(["retrieval", "src.retrieval"], "retrieve_relevant_chunks")
generate_rag_response = _optional_import(["llm_integration", "src.llm_integration"], "generate_rag_response")
generate_response = _optional_import(["llm_integration", "src.llm_integration"], "generate_response")


def _ensure_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("interactions", [])


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


def _retrieve_context(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if retrieve_context:
        return retrieve_context(question, top_k=top_k)
    if retrieve_relevant_chunks:
        return retrieve_relevant_chunks(question, top_k=top_k)
    return []


def _generate_answer(question: str, context: List[Dict[str, Any]]) -> Dict[str, Any]:
    if generate_rag_response:
        result = generate_rag_response(question, context)
    elif generate_response:
        result = generate_response(question, context)
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
        st.header("Configurações")
        st.write("Esta tela exibe o chat e integra busca e resposta quando disponíveis.")

        if st.button("Limpar conversa", use_container_width=True):
            st.session_state.messages = []
            st.session_state.interactions = []
            st.rerun()

        st.divider()
        st.subheader("Estado")
        st.write(f"Mensagens: {len(st.session_state.messages)}")
        st.write(f"Interações: {len(st.session_state.interactions)}")

    col_main, col_side = st.columns([3, 1])

    with col_side:
        st.subheader("Fluxo")
        st.write("A interface consulta o contexto e exibe a resposta gerada.")

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

