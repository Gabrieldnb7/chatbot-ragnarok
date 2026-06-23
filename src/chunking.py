import hashlib
import json
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Quantidade máxima de caracteres em cada chunk.
CHUNK_SIZE = 1000

# Parte repetida entre chunks consecutivos para preservar o contexto quando uma informação fica próxima da divisão do texto.
CHUNK_OVERLAP = 200

# Modelo usado apenas como apoio interno para detectar mudança semântica entre trechos.
SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"

# Quanto menor a similaridade entre trechos consecutivos, maior a chance de mudança de assunto.
SEMANTIC_SIMILARITY_THRESHOLD = 0.45

# Evita criar chunks pequenos demais só porque duas frases tiveram baixa similaridade.
MIN_SEMANTIC_CHUNK_SIZE = 300

_SENTENCE_SEPARATOR_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\[])")
_PARAGRAPH_SEPARATOR_PATTERN = re.compile(r"\n\s*\n+")

# Abreviações conhecidas do português que NÃO devem ser confundidas com fim de frase.
# O split por sentenças protege estas abreviações temporariamente para evitar
# que o "." seja interpretado como pontuação de encerramento.
_ABREVIACOES_CONHECIDAS = re.compile(
    r"\b(?:Dr|Sr|Sra|Srta|Srtas|Srs|Sras|etc|Ltda|obs|art|pág|vol|cap|ed|Ex|V)\.\s",
    re.UNICODE,
)
# Placeholder usado durante a proteção (caractere de controle não-imprimível).
# Durante a proteção, substitui o espaço após a abreviatura para que o
# separador de sentenças não confunda "Dr. " com fim de frase.
# Na restauração, o marcador é convertido de volta para um espaço simples.
_ABBREV_MARKER = "\x00"

_semantic_model = None


def _get_semantic_model():
    """Carrega o modelo semântico apenas quando ele for necessário."""
    global _semantic_model

    if _semantic_model is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Não foi possível importar sentence_transformers. "
                "Execute: pip install -r requirements.txt"
            ) from exc

        _semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    return _semantic_model


def _generate_doc_id(metadata: dict, cleaned_text: str) -> str:
    """Gera um identificador estável a partir dos metadados e do conteúdo do documento."""
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
    """Divide trechos grandes demais usando o RecursiveCharacterTextSplitter como fallback."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator="start",
    )
    return [chunk.strip() for chunk in text_splitter.split_text(text) if chunk.strip()]


def _protect_abbreviations(text: str) -> str:
    """Protege abreviações conhecidas para o separador de sentenças não quebrá-las.

    Remove o espaço após a abreviatura e insere um marcador temporário no lugar.
    Ex: "Dr. Carlos" → "Dr.\\x00Carlos"

    Após o split por sentenças, _restore_abbreviations() converte o marcador
    de volta para um espaço simples, restaurando o texto original sem duplicação.
    """
    def _protect(match: re.Match) -> str:
        # match.group(0) é "Dr. " — remove o espaço final e coloca o marcador
        return match.group(0).rstrip() + _ABBREV_MARKER
    return _ABREVIACOES_CONHECIDAS.sub(_protect, text)


def _restore_abbreviations(text: str) -> str:
    """Restaura os espaços removidos pela proteção de abreviações."""
    return text.replace(_ABBREV_MARKER, " ")


def _split_into_semantic_units(cleaned_text: str) -> list:
    """
    Quebra o texto em unidades de comparação semântica.

    Primeiro o texto é dividido por parágrafos e frases. Depois essas unidades
    são comparadas por embeddings temporários para decidir onde os chunks devem
    começar e terminar.

    Abreviações conhecidas (Dr., Sr., Sra., etc.) são protegidas antes do
    split por sentenças para evitar que o ponto final seja confundido com
    pontuação de encerramento de frase.
    """
    paragraphs = [
        paragraph.strip()
        for paragraph in _PARAGRAPH_SEPARATOR_PATTERN.split(cleaned_text)
        if paragraph.strip()
    ]

    semantic_units = []

    for paragraph in paragraphs:
        # Protege abreviações antes de dividir em sentenças
        paragraph_protected = _protect_abbreviations(paragraph)

        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_SEPARATOR_PATTERN.split(paragraph_protected)
            if sentence.strip()
        ]

        if not sentences:
            continue

        for sentence in sentences:
            # Restaura abreviações em cada sentença
            sentence = _restore_abbreviations(sentence)

            if len(sentence) <= CHUNK_SIZE:
                semantic_units.append(sentence)
            else:
                semantic_units.extend(_split_long_text(sentence))

    return semantic_units


def _encode_semantic_units(semantic_units: list):
    """
    Gera embeddings temporários(sem salvar nem retornar eles) apenas para comparar a semântica dos trechos.
    """
    model = _get_semantic_model()
    return model.encode(
        semantic_units,
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True,
    )


def _join_units_by_indexes(semantic_units: list, indexes: list) -> str:
    """Junta unidades de texto preservando separação legível entre elas."""
    return "\n\n".join(semantic_units[index] for index in indexes).strip()


def _cosine_similarity(normalized_embeddings, first_index: int, second_index: int) -> float:
    """Calcula similaridade de cosseno entre dois embeddings já normalizados."""
    return float(normalized_embeddings[first_index] @ normalized_embeddings[second_index])


def _get_overlap_indexes(current_indexes: list, semantic_units: list) -> list:
    """Mantém overlap reaproveitando unidades completas, sem começar no meio da frase."""
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
    """Monta chunks usando mudança semântica e limite máximo de tamanho."""
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

        # Regra 1: nunca passar do tamanho máximo definido para o chunk.
        if len(candidate_text) > CHUNK_SIZE:
            texts.append(current_text)
            current_indexes = _get_overlap_indexes(current_indexes, semantic_units)

            candidate_text = _join_units_by_indexes(semantic_units, current_indexes + [index])
            if len(candidate_text) > CHUNK_SIZE:
                current_indexes = []

        # Regra 2: se houve queda semântica relevante, abre um novo chunk.
        if current_indexes:
            current_text = _join_units_by_indexes(semantic_units, current_indexes)

            # Calcula a similaridade entre o CENTROIDE do chunk atual e a nova
            # unidade. Usar a média (centroide) de todos os embeddings do chunk
            # é mais estável do que comparar apenas com a última unidade, pois
            # evita que uma unidade de transição genérica (ex: "Siga as
            # instruções acima.") cause uma quebra falsa no chunk.
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

    Parâmetros:
        cleaned_text (str): Texto limpo e anonimizado recebido da issue do Felipe.
        metadata (dict): Metadados do documento, como título e fonte original.

    Retorno:
        list: Lista de dicionários com id, doc_id, texto e metadata.
    """
    if not isinstance(cleaned_text, str):
        raise TypeError("cleaned_text deve ser uma string.")

    if not isinstance(metadata, dict):
        raise TypeError("metadata deve ser um dicionário.")

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


# Teste isolado da Tarefa 02.
if __name__ == "__main__":
    # SIMULAÇÃO 1:

    exemplo_texto_limpo_1 = """
    Procedimento de recuperação de acesso ao sistema institucional. A usuária
    [NOME REMOVIDO], inscrita no CPF [CPF REMOVIDO] e cadastrada com o e-mail
    [EMAIL REMOVIDO], informou que não consegue entrar no sistema de atendimento.
    Para iniciar a recuperação, o usuário deve acessar a página oficial de
    autenticação e selecionar a opção Esqueci minha senha. Em seguida, deve
    informar o e-mail institucional cadastrado e aguardar o envio da mensagem
    com o link temporário de redefinição. O link deve ser utilizado dentro do
    prazo informado na mensagem. Caso o e-mail não seja recebido, o usuário
    deve verificar as pastas de spam e lixo eletrônico. Se a mensagem continuar
    ausente, deve confirmar se o endereço institucional foi digitado corretamente.

    Depois da redefinição, a nova senha deve respeitar os critérios exibidos na
    tela, incluindo tamanho mínimo e combinação de letras, números e caracteres
    especiais. Quando a conta estiver bloqueada por excesso de tentativas, o
    usuário deve aguardar o período indicado antes de tentar novamente. Se o
    bloqueio permanecer, o chamado deve ser encaminhado ao atendimento
    responsável por acessos. O chamado deve informar o sistema utilizado, a
    mensagem de erro e o horário aproximado da tentativa, sem incluir senha ou
    outros dados sensíveis.

    Em erros de autenticação após a troca da senha, recomenda-se encerrar todas
    as sessões abertas, fechar o navegador, abri-lo novamente e repetir o acesso.
    Caso o sistema utilize autenticação em dois fatores, o usuário deve confirmar
    se o dispositivo cadastrado está disponível e com data e hora corretas.
    Códigos temporários expirados não devem ser reutilizados. Quando houver troca
    ou perda do dispositivo autenticador, a recuperação deverá seguir o
    procedimento institucional e poderá exigir validação humana.

    O atendente deve utilizar somente documentos oficiais da base de conhecimento.
    Quando os documentos recuperados não apresentarem evidência suficiente, o
    caso deverá ser encaminhado para triagem humana. A fonte original deverá
    permanecer associada aos trechos para que as próximas etapas consigam informar
    qual documento fundamentou a resposta.
    """

    # SIMULAÇÃO 2:
    # Este texto usa os MESMOS metadados do primeiro, mas possui conteúdo diferente.
    # Só tá aqui pra provar que vai gerar ids diferentes e resolveu o problema que achamos na call
    exemplo_texto_limpo_2 = """
    Procedimento de recuperação de acesso ao sistema institucional. O usuário
    informou que consegue acessar o sistema, mas não consegue validar o segundo
    fator de autenticação. A primeira orientação é confirmar se o aplicativo
    autenticador está instalado no dispositivo correto e se a data e hora do
    aparelho estão configuradas automaticamente.

    Caso o código temporário seja recusado, o usuário deve gerar um novo código
    e tentar novamente dentro do prazo de validade. Códigos antigos ou já
    utilizados não devem ser reaproveitados. Se o usuário tiver trocado de
    celular, perdido o dispositivo ou removido o aplicativo autenticador, o caso
    deve seguir o procedimento institucional de recuperação de segundo fator.

    Quando a recuperação automática não estiver disponível, o chamado deve ser
    encaminhado para triagem humana. O atendente deve registrar o sistema
    afetado, a mensagem exibida na tela e o horário aproximado da tentativa,
    sem solicitar senha, código temporário ou dados sensíveis do usuário.
    """

    metadados_de_teste = {
        "titulo": "Procedimento de recuperação de acesso",
        "fonte": "Manual interno de suporte - acesso institucional",
    }

    print("TESTE ISOLADO DA MINHA ISSUE 2")
    print(
        "Este teste usa exemplos hipotéticos de textos sintéticos já limpos e anonimizados só pra testar se deu bom."
    )
    print(
        "Esse teste não executa nem recebe diretamente a Issue 1."
    )
    print(
        "Na integração final do projeto, para usar o texto REAL final, o parâmetro cleaned_text da minha issue receberá "
        "diretamente o texto retornado por ingest_and_anonymize() da issue do Felipe.\n"
    )
    print(
        "OBS: usei o all-MiniLM-L6-v2 só pra dividir os chuncks com base em semantica como me sugeriram. "
        "Mas não salvei nem retornei os embeddings para não invadir as tarefas da issue 3.\n"
    )

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
    print("Os dois documentos abaixo usam o MESMO título e a MESMA fonte.")
    print("Mesmo assim, como o conteúdo é diferente, os doc_ids também devem ser diferentes.\n")

    print(f"DOC_ID DO DOCUMENTO 1: {doc_id_1}")
    print(f"DOC_ID DO DOCUMENTO 2: {doc_id_2}")

    if doc_id_1 != doc_id_2:
        print("RESULTADO: OK - Os doc_ids ficaram diferentes mesmo com título/fonte iguais.\n")
    else:
        print("RESULTADO: ERRO - Os doc_ids ficaram iguais. Isso indicaria risco de colisão.\n")

    print(f"A função chunk_document() gerou {len(resultado_1)} chunks para o documento 1.\n")

    for chunk in resultado_1:
        print(f"ID DO CHUNK: {chunk['id']}")
        print(f"DOC_ID: {chunk['doc_id']}")
        print(f"METADATA: {chunk['metadata']}")
        print(f"TEXTO ({len(chunk['texto'])} caracteres):")
        print(chunk["texto"])
        print("-" * 70)

    print(f"\nA função chunk_document() gerou {len(resultado_2)} chunks para o documento 2.\n")

    for chunk in resultado_2:
        print(f"ID DO CHUNK: {chunk['id']}")
        print(f"DOC_ID: {chunk['doc_id']}")
        print(f"METADATA: {chunk['metadata']}")
        print(f"TEXTO ({len(chunk['texto'])} caracteres):")
        print(chunk["texto"])
        print("-" * 70)