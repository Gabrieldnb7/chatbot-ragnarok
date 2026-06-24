import hashlib
import json
import re

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    class RecursiveCharacterTextSplitter:
        def __init__(
            self,
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=None,
            keep_separator="start",
        ):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
            self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]
            self.keep_separator = keep_separator

        def split_text(self, text):
            text = (text or "").strip()
            if not text:
                return []

            chunks = []
            start = 0
            text_length = self.length_function(text)

            while start < text_length:
                end = min(start + self.chunk_size, text_length)
                if end < text_length:
                    window = text[start:end]
                    split_at = -1
                    for sep in self.separators:
                        if not sep:
                            continue
                        pos = window.rfind(sep)
                        if pos > split_at:
                            split_at = pos + len(sep)
                    if 0 < split_at < len(window):
                        end = start + split_at

                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)

                if end >= text_length:
                    break

                start = max(0, end - self.chunk_overlap)
                if start == end:
                    start = end + 1

            return chunks


# Quantidade maxima de caracteres em cada chunk.
CHUNK_SIZE = 1000

# Parte repetida entre chunks consecutivos para preservar contexto perto das quebras.
CHUNK_OVERLAP = 200

# Modelo usado apenas como apoio interno para detectar mudanca semantica entre trechos.
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

# Quanto menor a similaridade entre trechos consecutivos, maior a chance de mudanca de assunto.
SEMANTIC_SIMILARITY_THRESHOLD = 0.45

# Evita criar chunks pequenos demais so porque duas frases tiveram baixa similaridade.
MIN_SEMANTIC_CHUNK_SIZE = 300

_SENTENCE_SEPARATOR_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[])")
_PARAGRAPH_SEPARATOR_PATTERN = re.compile(r"\n\s*\n+")

# Abreviacoes conhecidas do portugues que nao devem ser confundidas com fim de frase.
_ABREVIACOES_CONHECIDAS = re.compile(
    r"\b(?:Dr|Sr|Sra|Srta|Srtas|Srs|Sras|etc|Ltda|obs|art|pag|vol|cap|ed|Ex|V)\.\s",
    re.UNICODE,
)

# Placeholder temporario usado para proteger abreviacoes durante o split por sentencas.
_ABBREV_MARKER = "\x00"

_semantic_model = None


def _get_semantic_model():
    """Carrega o modelo semantico apenas quando ele for necessario."""
    global _semantic_model

    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Nao foi possivel importar sentence_transformers. "
                "Execute: pip install -r requirements.txt"
            ) from exc

        _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    return _semantic_model


def _generate_doc_id(metadata: dict, cleaned_text: str) -> str:
    """Gera um identificador estavel a partir dos metadados e do conteudo."""
    metadata_serialized = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )

    hash_metadata = hashlib.sha256(metadata_serialized.encode("utf-8")).hexdigest()
    hash_content = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
    hash_document = hashlib.sha256(
        f"{hash_metadata}:{hash_content}".encode("utf-8")
    ).hexdigest()

    return f"doc_{hash_document[:12]}"


def _split_long_text(text: str) -> list:
    """Divide trechos grandes demais usando RecursiveCharacterTextSplitter como fallback."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator="start",
    )
    # O separador ajuda na quebra, mas a pontuacao pertence ao chunk anterior.
    return [
        chunk.lstrip(".!?;:,").strip()
        for chunk in text_splitter.split_text(text)
        if chunk.strip()
    ]


def _protect_abbreviations(text: str) -> str:
    """Protege abreviacoes conhecidas para o separador de sentencas nao quebra-las."""

    def _protect(match: re.Match) -> str:
        return match.group(0).rstrip() + _ABBREV_MARKER

    return _ABREVIACOES_CONHECIDAS.sub(_protect, text)


def _restore_abbreviations(text: str) -> str:
    """Restaura os espacos removidos pela protecao de abreviacoes."""
    return text.replace(_ABBREV_MARKER, " ")


def _split_into_semantic_units(cleaned_text: str) -> list:
    """
    Quebra o texto em unidades de comparacao semantica.

    Primeiro o texto e dividido por paragrafos e frases. Depois essas unidades
    sao comparadas por embeddings temporarios para decidir onde os chunks devem
    comecar e terminar.
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_SEPARATOR_PATTERN.split(cleaned_text)
        if paragraph.strip()
    ]

    semantic_units = []

    for paragraph in paragraphs:
        paragraph_protected = _protect_abbreviations(paragraph)

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SEPARATOR_PATTERN.split(paragraph_protected)
            if sentence.strip()
        ]

        if not sentences:
            continue

        for sentence in sentences:
            sentence = _restore_abbreviations(sentence)

            if len(sentence) <= CHUNK_SIZE:
                semantic_units.append(sentence)
            else:
                semantic_units.extend(_split_long_text(sentence))

    return semantic_units


def _encode_semantic_units(semantic_units: list):
    """Gera embeddings temporarios apenas para comparar a semantica dos trechos."""
    model = _get_semantic_model()
    return model.encode(
        semantic_units,
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True,
    )


def _join_units_by_indexes(semantic_units: list, indexes: list) -> str:
    """Junta unidades de texto preservando separacao legivel entre elas."""
    return "\n\n".join(semantic_units[index] for index in indexes).strip()


def _cosine_similarity(normalized_embeddings, first_index: int, second_index: int) -> float:
    """Calcula similaridade de cosseno entre dois embeddings ja normalizados."""
    return float(normalized_embeddings[first_index] @ normalized_embeddings[second_index])


def _get_overlap_indexes(current_indexes: list, semantic_units: list) -> list:
    """Mantem overlap reaproveitando unidades completas, sem comecar no meio da frase."""
    overlap_indexes = []
    overlap_size = 0

    for index in reversed(current_indexes):
        unit_size = len(semantic_units[index]) + (2 if overlap_indexes else 0)

        if overlap_indexes and overlap_size + unit_size > CHUNK_OVERLAP:
            break

        overlap_indexes.insert(0, index)
        overlap_size += unit_size

        if overlap_size >= CHUNK_OVERLAP:
            break

    return overlap_indexes


def _build_semantic_chunks(semantic_units: list) -> list:
    """Monta chunks usando mudanca semantica e limite maximo de tamanho."""
    if not semantic_units:
        return []

    if len(semantic_units) == 1:
        return _split_long_text(semantic_units[0])

    embeddings = _encode_semantic_units(semantic_units)
    texts = []
    current_indexes = []

    for index, unit in enumerate(semantic_units):
        if not current_indexes:
            current_indexes.append(index)
            continue

        current_text = _join_units_by_indexes(semantic_units, current_indexes)
        candidate_text = _join_units_by_indexes(semantic_units, current_indexes + [index])

        if len(candidate_text) > CHUNK_SIZE:
            texts.append(current_text)
            current_indexes = _get_overlap_indexes(current_indexes, semantic_units)

            candidate_text = _join_units_by_indexes(semantic_units, current_indexes + [index])
            if len(candidate_text) > CHUNK_SIZE:
                current_indexes = []

        if current_indexes:
            current_text = _join_units_by_indexes(semantic_units, current_indexes)
            centroid = embeddings[current_indexes].mean(axis=0)
            similarity = float(centroid @ embeddings[index])

            if (
                len(current_text) >= MIN_SEMANTIC_CHUNK_SIZE
                and similarity < SEMANTIC_SIMILARITY_THRESHOLD
            ):
                texts.append(current_text)
                current_indexes = _get_overlap_indexes(current_indexes, semantic_units)

                candidate_text = _join_units_by_indexes(semantic_units, current_indexes + [index])
                if len(candidate_text) > CHUNK_SIZE:
                    current_indexes = []

        current_indexes.append(index)

    if current_indexes:
        texts.append(_join_units_by_indexes(semantic_units, current_indexes))

    return texts


def chunk_document(cleaned_text: str, metadata: dict) -> list:
    """
    Divide um texto limpo em chunks menores e associa os metadados do documento.

    Parametros:
        cleaned_text (str): Texto limpo e anonimizado.
        metadata (dict): Metadados do documento, como titulo e fonte original.

    Retorno:
        list: Lista de dicionarios com id, doc_id, texto e metadata.
    """
    if not isinstance(cleaned_text, str):
        raise TypeError("cleaned_text deve ser uma string.")

    if not isinstance(metadata, dict):
        raise TypeError("metadata deve ser um dicionario.")

    cleaned_text = cleaned_text.strip()

    if not cleaned_text:
        return []

    semantic_units = _split_into_semantic_units(cleaned_text)
    texts = _build_semantic_chunks(semantic_units)
    doc_id = _generate_doc_id(metadata, cleaned_text)

    chunks = []

    for position, text in enumerate(texts, start=1):
        chunks.append(
            {
                "id": f"{doc_id}_chunk_{position:04d}",
                "doc_id": doc_id,
                "texto": text,
                "metadata": metadata.copy(),
            }
        )

    return chunks


if __name__ == "__main__":
    exemplo_texto_limpo_1 = """
    Procedimento de recuperacao de acesso ao sistema institucional. A usuaria
    [NOME REMOVIDO], inscrita no CPF [CPF REMOVIDO] e cadastrada com o e-mail
    [EMAIL REMOVIDO], informou que nao consegue entrar no sistema de atendimento.
    Para iniciar a recuperacao, o usuario deve acessar a pagina oficial de
    autenticacao e selecionar a opcao Esqueci minha senha. Em seguida, deve
    informar o e-mail institucional cadastrado e aguardar o envio da mensagem
    com o link temporario de redefinicao.

    Depois da redefinicao, a nova senha deve respeitar os criterios exibidos na
    tela, incluindo tamanho minimo e combinacao de letras, numeros e caracteres
    especiais. Quando a conta estiver bloqueada por excesso de tentativas, o
    usuario deve aguardar o periodo indicado antes de tentar novamente.
    """

    exemplo_texto_limpo_2 = """
    Procedimento de recuperacao de acesso ao sistema institucional. O usuario
    informou que consegue acessar o sistema, mas nao consegue validar o segundo
    fator de autenticacao. A primeira orientacao e confirmar se o aplicativo
    autenticador esta instalado no dispositivo correto e se a data e hora do
    aparelho estao configuradas automaticamente.
    """

    metadados_de_teste = {
        "titulo": "Procedimento de recuperacao de acesso",
        "fonte": "Manual interno de suporte - acesso institucional",
    }

    print("TESTE ISOLADO DA TAREFA 02")
    print("Este teste usa textos sinteticos ja limpos e anonimizados.\n")

    resultado_1 = chunk_document(
        cleaned_text=exemplo_texto_limpo_1,
        metadata=metadados_de_teste,
    )
    resultado_2 = chunk_document(
        cleaned_text=exemplo_texto_limpo_2,
        metadata=metadados_de_teste,
    )

    doc_id_1 = resultado_1[0]["doc_id"] if resultado_1 else None
    doc_id_2 = resultado_2[0]["doc_id"] if resultado_2 else None

    print("TESTE DE UNICIDADE DO DOC_ID")
    print("Os dois documentos usam o mesmo titulo e a mesma fonte.")
    print("Como o conteudo e diferente, os doc_ids tambem devem ser diferentes.\n")

    print(f"DOC_ID DO DOCUMENTO 1: {doc_id_1}")
    print(f"DOC_ID DO DOCUMENTO 2: {doc_id_2}")

    if doc_id_1 != doc_id_2:
        print("RESULTADO: OK - Os doc_ids ficaram diferentes.\n")
    else:
        print("RESULTADO: ERRO - Os doc_ids ficaram iguais.\n")

    print(f"A funcao chunk_document() gerou {len(resultado_1)} chunks para o documento 1.\n")

    for chunk in resultado_1:
        print(f"ID DO CHUNK: {chunk['id']}")
        print(f"DOC_ID: {chunk['doc_id']}")
        print(f"METADATA: {chunk['metadata']}")
        print(f"TEXTO ({len(chunk['texto'])} caracteres):")
        print(chunk["texto"])
        print("-" * 70)

    print(f"\nA funcao chunk_document() gerou {len(resultado_2)} chunks para o documento 2.\n")

    for chunk in resultado_2:
        print(f"ID DO CHUNK: {chunk['id']}")
        print(f"DOC_ID: {chunk['doc_id']}")
        print(f"METADATA: {chunk['metadata']}")
        print(f"TEXTO ({len(chunk['texto'])} caracteres):")
        print(chunk["texto"])
        print("-" * 70)
