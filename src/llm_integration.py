# Tarefa 06: Orquestração do RAG e Integração com LLM

from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Sequence

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependencia configuracional
    load_dotenv = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI as _ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - permite fallback sem dependencia instalada
    _ChatGoogleGenerativeAI = None

from stopwordsiso import stopwords as _iso_stopwords

DEFAULT_SCORE_THRESHOLD = 0.12
MAX_EXCERPT_LENGTH = 320
MAX_SENTENCES_PER_SUMMARY = 8
MAX_FACTS_PER_GROUP = 8
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"

if load_dotenv is not None:
    load_dotenv()
DOMAIN_TERMS = {
    "api",
    "apis",
    "fastapi",
    "git",
    "github",
    "gitlab",
    "json",
    "moodle",
    "pydantic",
    "python",
    "rest",
    "sql",
}


def _load_stopwords() -> set:
    return set(_iso_stopwords("pt")) - DOMAIN_TERMS


STOPWORDS = _load_stopwords()


def _gemini_config() -> Dict[str, str]:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or "SuaChave" in api_key:
        return {}
    return {
        "api_key": api_key,
        "model": os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL,
    }


def _build_provider_prompt(query: str, retrieved_context: Sequence[Dict[str, Any]]) -> str:
    context_lines = []
    for index, chunk in enumerate(retrieved_context, start=1):
        source_ref = _format_source_ref(chunk)
        source_label = _format_source_label(chunk)
        source_origin = _format_source_origin(chunk)
        source_page = _format_source_page(chunk)
        page_part = f", {source_page}" if source_page else ""
        excerpt = _format_context_excerpt(chunk, 900)
        context_lines.append(
            f"[{index}] [Fonte: {source_ref}] {source_label} ({source_origin}{page_part})\n{excerpt}"
        )

    context_block = "\n\n".join(context_lines)
    return (
        "Você é um assistente RAG para análise de documentos. "
        "Responda em português brasileiro padrão. "
        "Use exclusivamente o CONTEXTO RECUPERADO abaixo. "
        "Se a resposta não estiver sustentada pelo contexto, "
        "NÃO invente nada — responda que não pode ajudar com essa "
        "pergunta no momento e sugira procurar ajuda humana. "
        "Não invente informações, datas, valores, nomes, tecnologias ou conclusões. "
        "Cite as fontes usadas no formato [Fonte: id].\n\n"
        f"PERGUNTA DO USUÁRIO:\n{query}\n\n"
        f"CONTEXTO RECUPERADO:\n{context_block}\n\n"
        "RESPOSTA:"
    )


def _call_gemini_llm(query: str, retrieved_context: Sequence[Dict[str, Any]]) -> str:
    config = _gemini_config()
    if not config or _ChatGoogleGenerativeAI is None:
        return ""

    prompt = _build_provider_prompt(query, retrieved_context)
    try:
        llm = _ChatGoogleGenerativeAI(
            google_api_key=config["api_key"],
            model=config["model"],
            temperature=0.1,
            max_output_tokens=900,
        )
        response = llm.invoke([
            (
                "system",
                "Você é um assistente RAG. Responda SOMENTE se o contexto abaixo tiver informação suficiente. "
                "Caso contrário, diga claramente que não pode ajudar com essa pergunta e sugira "
                "procurar ajuda humana. Não invente respostas. Cite fontes no formato [Fonte: id].",
            ),
            ("human", prompt),
        ])
        return str(getattr(response, "content", response)).strip()
    except Exception:
        return ""


def _call_configured_llm(query: str, retrieved_context: Sequence[Dict[str, Any]]) -> tuple[str, str]:
    gemini_answer = _call_gemini_llm(query, retrieved_context)
    if gemini_answer:
        return gemini_answer, "gemini"

    return "", "local"

WORD_RE = re.compile(r"\b[\wÀ-ÿ]+\b", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
DATE_RE = re.compile(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b")
MONEY_RE = re.compile(r"(?:R\$\s?\d{1,3}(?:\.\d{3})*(?:,\d{2})?|\$\s?\d+(?:\.\d{2})?)")
PERCENT_RE = re.compile(r"\b\d+(?:,\d+)?%\b")
NUMBER_RE = re.compile(r"\b\d+(?:[\.,]\d+)?\b")
ACTION_RE = re.compile(
    r"\b(deve|deverá|devera|precisa|necessita|solicita|solicitado|aprovado|indeferido|autorizado|proibido|prazo|vencimento|valor|data|responsável|responsavel|obrigatório|obrigatorio)\b",
    re.IGNORECASE,
)


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    normalized = _normalize_text(text).lower()
    return [token for token in WORD_RE.findall(normalized) if token and token not in STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(normalized) if sentence.strip()]
    if sentences:
        return sentences
    return [normalized]


def _format_source_label(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("titulo") or "Documento"
    return "Documento"


def _format_source_ref(chunk: Dict[str, Any]) -> str:
    return str(chunk.get("id") or chunk.get("doc_id") or _format_source_label(chunk))


def _format_source_origin(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("fonte") or "documento"
    return "documento"


def _format_source_page(chunk: Dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    if isinstance(metadata, dict):
        page_number = metadata.get("page_number")
        if page_number is not None:
            return f"página {page_number}"
    return ""


def _format_score(chunk: Dict[str, Any]) -> str:
    score = chunk.get("score")
    if isinstance(score, (int, float)):
        return f"{float(score):.4f}"
    return "n/d"


def _format_context_excerpt(chunk: Dict[str, Any], max_length: int = MAX_EXCERPT_LENGTH) -> str:
    text = (chunk.get("texto", "") or "").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _collect_relevant_terms(query: str, retrieved_context: Sequence[Dict[str, Any]]) -> List[str]:
    query_terms = set(_tokenize(query))
    counter: Counter[str] = Counter()
    for chunk in retrieved_context:
        counter.update(_tokenize(str(chunk.get("texto", ""))))
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            counter.update(_tokenize(str(metadata.get("titulo", ""))))
            counter.update(_tokenize(str(metadata.get("fonte", ""))))
        counter.update(chunk.get("match_terms", []))
    terms = [term for term, _count in counter.most_common() if term not in query_terms and len(term) > 2 and not term.isdigit()]
    return terms[:6]


def _summarize_chunks(retrieved_context: Sequence[Dict[str, Any]]) -> str:
    sentences: List[str] = []
    for chunk in retrieved_context:
        text = (chunk.get("texto", "") or "").strip()
        if not text:
            continue
        chunk_sentences = _split_sentences(text)
        if chunk_sentences:
            sentences.append(chunk_sentences[0])
    unique_sentences = []
    seen = set()
    for sentence in sentences:
        normalized = sentence.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_sentences.append(sentence)
        if len(unique_sentences) >= MAX_SENTENCES_PER_SUMMARY:
            break
    return " ".join(unique_sentences)


def _collect_document_text(retrieved_context: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for chunk in retrieved_context:
        text = (chunk.get("texto", "") or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _extract_key_facts(text: str) -> Dict[str, List[str]]:
    facts = {
        "dates": list(dict.fromkeys(DATE_RE.findall(text)))[:MAX_FACTS_PER_GROUP],
        "money": list(dict.fromkeys(MONEY_RE.findall(text)))[:MAX_FACTS_PER_GROUP],
        "percentages": list(dict.fromkeys(PERCENT_RE.findall(text)))[:MAX_FACTS_PER_GROUP],
        "numbers": list(dict.fromkeys(NUMBER_RE.findall(text)))[:MAX_FACTS_PER_GROUP],
    }
    action_sentences = []
    for sentence in _split_sentences(text):
        if ACTION_RE.search(sentence):
            action_sentences.append(sentence.strip())
        if len(action_sentences) >= MAX_FACTS_PER_GROUP:
            break
    facts["actions"] = action_sentences
    return facts


BULLET_RE = re.compile(r"^\s*(?:[\u25cf\u2022*\-]|\d+[.)])\s+")


def _clean_list_item(line: str) -> str:
    line = re.sub(r"\s+", " ", line or "").strip()
    line = BULLET_RE.sub("", line).strip()
    return line.rstrip(";.")


def _extract_section_insights(retrieved_context: Sequence[Dict[str, Any]]) -> List[str]:
    sections: Dict[str, Dict[str, Any]] = {}
    for chunk in retrieved_context:
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            section = metadata.get("section_path") or metadata.get("section_title") or "Documento"
        else:
            section = "Documento"
        bucket = sections.setdefault(section, {"text": [], "items": []})
        text = (chunk.get("texto", "") or "").strip()
        if not text:
            continue
        bucket["text"].append(text)
        for raw_line in text.splitlines():
            item = _clean_list_item(raw_line)
            if not item or len(item) < 4:
                continue
            normalized = _normalize_text(item).lower()
            if normalized == _normalize_text(section).lower():
                continue
            is_bullet = bool(BULLET_RE.match(raw_line))
            is_relevant_line = is_bullet or ACTION_RE.search(item) or normalized in {
                "python",
                "fastapi",
                "pydantic",
                "sql",
                "git",
                "github",
                "gitlab",
                "rest api",
                "json",
                "moodle",
            }
            if is_relevant_line and item not in bucket["items"]:
                bucket["items"].append(item)

    insights: List[str] = []
    for section, data in sections.items():
        items = data["items"][:6]
        if not items:
            combined = " ".join(data["text"])
            items = _split_sentences(combined)[:2]
        if items:
            insights.append(f"{section}: " + "; ".join(items))
        if len(insights) >= 8:
            break
    return insights


def _select_supporting_sentences(query: str, retrieved_context: Sequence[Dict[str, Any]]) -> List[str]:
    query_terms = set(_tokenize(query))
    sentences: List[str] = []
    for chunk in retrieved_context:
        for sentence in _split_sentences(chunk.get("texto", "")):
            sentence_terms = set(_tokenize(sentence))
            if not sentence_terms:
                continue
            if query_terms & sentence_terms or ACTION_RE.search(sentence):
                sentences.append(sentence)
            if len(sentences) >= MAX_SENTENCES_PER_SUMMARY:
                return sentences
    return sentences


def _build_document_summary(query: str, retrieved_context: Sequence[Dict[str, Any]]) -> str:
    if not retrieved_context:
        return "Não encontrei um resumo confiável porque não havia trechos suficientes para análise."

    top_titles: List[str] = []
    pages: List[str] = []
    for chunk in retrieved_context[:4]:
        title = _format_source_label(chunk)
        if title not in top_titles:
            top_titles.append(title)
        page = _format_source_page(chunk)
        if page and page not in pages:
            pages.append(page)

    terms = _collect_relevant_terms(query, retrieved_context)
    summary_text = _summarize_chunks(retrieved_context)
    facts = _extract_key_facts(_collect_document_text(retrieved_context))
    section_insights = _extract_section_insights(retrieved_context)
    title_block = ", ".join(top_titles)
    topic_block = ", ".join(terms) if terms else "o conteúdo principal do documento"
    page_block = f" Páginas observadas: {', '.join(pages)}." if pages else ""

    fact_parts = []
    if facts["dates"]:
        fact_parts.append(f"datas encontradas: {', '.join(facts['dates'])}")
    if facts["money"]:
        fact_parts.append(f"valores encontrados: {', '.join(facts['money'])}")
    if facts["percentages"]:
        fact_parts.append(f"percentuais encontrados: {', '.join(facts['percentages'])}")
    if facts["actions"]:
        fact_parts.append(f"trechos de ação/decisão: {' | '.join(facts['actions'])}")
    if section_insights:
        fact_parts.append("estrutura do documento: " + " | ".join(section_insights[:4]))

    analytical_block = f" Informações identificadas: {'; '.join(fact_parts)}." if fact_parts else ""

    if summary_text:
        return (
            f"Os trechos mais relevantes vêm de {title_block}. "
            f"Pelos termos encontrados, o documento gira em torno de {topic_block}.{page_block}"
            f" Resumo do que foi localizado: {summary_text}.{analytical_block}"
        )
    return (
        f"Os trechos mais relevantes vêm de {title_block}. "
        f"Pelos termos encontrados, o documento gira em torno de {topic_block}.{page_block}{analytical_block}"
    )


def _build_search_overview(query: str, retrieved_context: Sequence[Dict[str, Any]]) -> str:
    if not retrieved_context:
        return f"Busquei por '{query}' na base de conhecimento, mas não encontrei evidências suficientes."

    sources = []
    for chunk in retrieved_context[:5]:
        label = _format_source_label(chunk)
        source_ref = _format_source_ref(chunk)
        source = _format_source_origin(chunk)
        page = _format_source_page(chunk)
        score = _format_score(chunk)
        match_terms = chunk.get("match_terms") or []
        page_part = f", {page}" if page else ""
        if match_terms:
            sources.append(f"[Fonte: {source_ref}] {label} ({source}{page_part}, score {score}, termos: {', '.join(match_terms[:4])})")
        else:
            sources.append(f"[Fonte: {source_ref}] {label} ({source}{page_part}, score {score})")

    source_block = "; ".join(sources)
    return f"Busquei a pergunta '{query}' priorizando os trechos mais próximos semanticamente e, em seguida, refinei com termos em comum. Fontes priorizadas: {source_block}."


def _build_analytical_answer(query: str, retrieved_context: List[Dict[str, Any]]) -> Dict[str, str]:
    search_overview = _build_search_overview(query, retrieved_context)
    document_summary = _build_document_summary(query, retrieved_context)
    supporting_sentences = _select_supporting_sentences(query, retrieved_context)
    facts = _extract_key_facts(_collect_document_text(retrieved_context))
    section_insights = _extract_section_insights(retrieved_context)
    llm_answer, llm_provider = _call_configured_llm(query, retrieved_context)

    source_lines = []
    for index, chunk in enumerate(retrieved_context, start=1):
        source_label = _format_source_label(chunk)
        source_ref = _format_source_ref(chunk)
        source_origin = _format_source_origin(chunk)
        source_page = _format_source_page(chunk)
        score_label = _format_score(chunk)
        excerpt = _format_context_excerpt(chunk)
        match_terms = chunk.get("match_terms") or []
        title_match_terms = chunk.get("title_match_terms") or []
        detail_bits = []
        if source_page:
            detail_bits.append(source_page)
        if match_terms:
            detail_bits.append(f"termos recuperados: {', '.join(match_terms[:5])}")
        if title_match_terms:
            detail_bits.append(f"correspondência no título: {', '.join(title_match_terms[:3])}")
        detail_block = f" | {'; '.join(detail_bits)}" if detail_bits else ""
        source_lines.append(
            f"- [Fonte: {source_ref}] {source_label} ({source_origin}{detail_block}, score {score_label})"
        )

    facts_block = []
    if facts["dates"]:
        facts_block.append(f"Datas/prazos: {', '.join(facts['dates'])}")
    if facts["money"]:
        facts_block.append(f"Valores: {', '.join(facts['money'])}")
    if facts["percentages"]:
        facts_block.append(f"Percentuais: {', '.join(facts['percentages'])}")
    if facts["actions"]:
        facts_block.append("Ações/decisões: " + " | ".join(facts["actions"]))
    if section_insights:
        facts_block.append("Análise por seções: " + " | ".join(section_insights))
    if not facts_block:
        facts_block.append("Não identifiquei datas, valores ou ações explícitas nos trechos recuperados.")

    # ── resposta_gerada concisa ──
    answer_parts = []
    if llm_answer:
        answer_parts.append(llm_answer)
    else:
        answer_parts.append(search_overview)
        if supporting_sentences:
            answer_parts.append("")
            answer_parts.append(" || ".join(supporting_sentences))

    answer_parts.append("")
    answer_parts.append("---")
    answer_parts.append("**Fontes consultadas:**")
    answer_parts.extend(source_lines)
    answer_parts.append("")
    answer_parts.append("---")
    answer_parts.append("*Resposta construída com base exclusivamente nos trechos recuperados, sem adicionar informações externas.*")

    return {
        "resposta_gerada": "\n".join(answer_parts),
        "resumo_busca": search_overview,
        "resumo_documento": document_summary,
        "analise_documento": "\n".join(facts_block),
        "usou_llm_externo": bool(llm_answer),
        "provedor_llm": llm_provider,
    }


def generate_rag_response(query: str, retrieved_context: list) -> dict:
    """Gera uma resposta fundamentada nos contextos recuperados."""
    if not isinstance(query, str):
        raise TypeError("query deve ser uma string.")
    if not isinstance(retrieved_context, list):
        raise TypeError("retrieved_context deve ser uma lista.")

    if not retrieved_context:
        search_overview = _build_search_overview(query, [])
        return {
            "resposta_gerada": (
                f"{search_overview}\n\n"
                "Não encontrei informações suficientes na minha base de conhecimento "
                "para responder a essa pergunta. "
                "Infelizmente, não posso ajudar com isso no momento.\n\n"
                "Por favor, procure ajuda humana para obter a informação correta."
            ),
            "resumo_busca": search_overview,
            "resumo_documento": "Não foi possível sintetizar um documento de referência porque nenhum trecho relevante foi recuperado.",
            "analise_documento": "Sem dados suficientes para an\u00e1lise.",
            "fontes": [],
            "usou_llm_externo": False,
            "provedor_llm": "local",
            "precisou_triagem": True,
            "confianca": 0.0,
        }

    top_score = max(float(chunk.get("score", 0.0)) for chunk in retrieved_context)
    search_overview = _build_search_overview(query, retrieved_context)
    document_summary = _build_document_summary(query, retrieved_context)

    if top_score < DEFAULT_SCORE_THRESHOLD:
        return {
            "resposta_gerada": (
                f"Pesquisa realizada:\n{search_overview}\n\n"
                f"Síntese do conteúdo:\n{document_summary}\n\n"
                "A evidência disponível é muito fraca para uma resposta confiável. "
                "Não posso ajudar com essa pergunta no momento.\n\n"
                "Por favor, procure ajuda humana ou inclua documentos mais específicos na base de conhecimento."
            ),
            "resumo_busca": search_overview,
            "resumo_documento": document_summary,
            "analise_documento": "Sem dados suficientes para an\u00e1lise.",
            "fontes": retrieved_context,
            "usou_llm_externo": False,
            "provedor_llm": "local",
            "precisou_triagem": True,
            "confianca": round(top_score, 4),
        }

    response_parts = _build_analytical_answer(query, retrieved_context)
    return {
        "resposta_gerada": response_parts["resposta_gerada"],
        "resumo_busca": response_parts["resumo_busca"],
        "resumo_documento": response_parts["resumo_documento"],
        "analise_documento": response_parts["analise_documento"],
        "fontes": retrieved_context,
        "usou_llm_externo": response_parts.get("usou_llm_externo", False),
        "provedor_llm": response_parts.get("provedor_llm", "local"),
        "precisou_triagem": False,
        "confianca": round(top_score, 4),
    }


def generate_response(query: str, retrieved_context: list) -> dict:
    """Alias compatível com a interface."""
    return generate_rag_response(query, retrieved_context)


if __name__ == "__main__":  # pragma: no cover - teste manual
    print(generate_rag_response("Como recuperar acesso?", []))
