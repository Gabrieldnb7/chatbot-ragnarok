import hashlib
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter


# Quantidade máxima de caracteres em cada chunk.
CHUNK_SIZE = 1000

# Parte repetida entre chunks consecutivos para preservar o contexto quando uma informação fica próxima da divisão do texto.
CHUNK_OVERLAP = 200


def _generate_doc_id(metadata: dict) -> str:
    """Gera um identificador estável para o documento a partir dos metadados."""
    metadata_serialized = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )
    hash_metadata = hashlib.sha256(metadata_serialized.encode("utf-8")).hexdigest()
    return f"doc_{hash_metadata[:12]}"


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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator="end",
    )

    texts = text_splitter.split_text(cleaned_text)
    doc_id = _generate_doc_id(metadata)

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
    # SIMULAÇÃO:

    exemplo_texto_limpo = """
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

    metadados_de_teste = {
        "titulo": "Procedimento de recuperação de acesso",
        "fonte": "Manual interno de suporte - acesso institucional",
    }

    print("TESTE ISOLADO DA MINHA ISSUE 2")
    print(
        "Este teste usa um EXEMPLO hipotético de texto sintético já limpo e anonimizado só pra testar se deu bom."
    )
    print(
        "Esse teste não executa nem recebe diretamente a Issue 1."
    )
    print(
        "Na integração final do projeto, para usar o texto REAL final, o parâmetro cleaned_text da minha issue receberá "
        "diretamente o texto retornado por ingest_and_anonymize() da issue do Felipe.\n"
    )

    resultado = chunk_document(
        cleaned_text=exemplo_texto_limpo,
        metadata=metadados_de_teste,
    )

    print(f"A função chunk_document() gerou {len(resultado)} chunks.\n")

    for chunk in resultado:
        print(f"ID DO CHUNK: {chunk['id']}")
        print(f"DOC_ID: {chunk['doc_id']}")
        print(f"METADATA: {chunk['metadata']}")
        print(f"TEXTO ({len(chunk['texto'])} caracteres):")
        print(chunk["texto"])
        print("-" * 70)