# Tarefa 01: Ingestão e anonimização

import re
import spacy
import nltk

# Configuração inicial: Baixar dependências do NLTK (executa apenas se não tiver)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Carregar o modelo de NLP em Português do spaCy
# (Usamos o 'sm' por ser mais leve, mas para maior precisão em produção, considere o 'lg')
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    raise OSError("Modelo do spaCy não encontrado. Execute: python -m spacy download pt_core_news_sm")

def ingest_and_anonymize(file_content: str) -> str:
    """
    Ingere um texto bruto, realiza limpeza estrutural e anonimiza dados sensíveis.
    
    Parâmetros:
    file_content (str): Texto bruto extraído do documento.
    
    Retorno:
    str: Texto limpo e com PIIs substituídos por placeholders.
    """
    if not isinstance(file_content, str):
        return ""

    # =================================================================
    # ETAPA 1: Limpeza Estrutural Básica (NLTK & Regex)
    # =================================================================
    # Remove quebras de linha excessivas, tabulações e espaços duplos
    texto_limpo = re.sub(r'\s+', ' ', file_content).strip()
    
    # Opcional: Se precisar padronizar frases de forma mais profunda, 
    # o NLTK pode ser usado para tokenização de sentenças, mas para 
    # anonimização, manter o texto como uma string única facilita o Regex.

    # =================================================================
    # ETAPA 2: Anonimização Determinística (Expressões Regulares)
    # Ideal para padrões bem definidos (CPFs, Cartões, Emails, Telefones)
    # =================================================================
    
    # Padrão para CPF (formato XXX.XXX.XXX-XX ou XXXXXXXXXXX)
    texto_limpo = re.sub(r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b', '[CPF REMOVIDO]', texto_limpo)
    
    # Padrão para E-mails
    texto_limpo = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REMOVIDO]', texto_limpo)
    
    # Padrão para Telefones Brasileiros (Ex: +55 (11) 98888-8888, 11 98888-8888)
    texto_limpo = re.sub(r'\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?\d{4,5}[-\s]?\d{4}\b', '[TELEFONE REMOVIDO]', texto_limpo)
    
    # Padrão para Cartões de Crédito (16 dígitos com ou sem separador)
    texto_limpo = re.sub(r'\b(?:\d{4}[-\s]?){3}\d{4}\b', '[DADO BANCARIO REMOVIDO]', texto_limpo)

    # =================================================================
    # ETAPA 3: Anonimização Contextual (spaCy - NER)
    # Ideal para Nomes, Organizações ou Entidades sem formato fixo
    # =================================================================
    
    doc = nlp(texto_limpo)
    texto_anonimizado = texto_limpo
    
    # Iteramos sobre as entidades encontradas em ordem reversa.
    # Fazemos isso de trás para frente para que a substituição de uma string
    # não altere os índices dos caracteres das entidades seguintes.
    for ent in reversed(doc.ents):
        # "PER" representa Pessoa no spaCy. 
        # Você também pode adicionar "ORG" (Organização) ou "LOC" (Local) se precisar.
        if ent.label_ == "PER":
            inicio = ent.start_char
            fim = ent.end_char
            texto_anonimizado = texto_anonimizado[:inicio] + "[NOME REMOVIDO]" + texto_anonimizado[fim:]

    return texto_anonimizado

# =================================================================
# TESTE DA FUNÇÃO
# =================================================================
if __name__ == "__main__":
    texto_bruto = """
    Documento de Transferência e Acordo.
    
    Eu, Carlos Eduardo da Silva, portador do CPF 123.456.789-00, concordo com os termos.
    Em caso de dúvidas, entrar em contato pelo e-mail carlos.silva@empresa.com.br ou 
    pelo telefone (11) 98765-4321.
    
    Pagamento efetuado com o cartão 4532 1122 3344 5566.
    Assinado em São Paulo.
    """
    
    resultado = ingest_and_anonymize(texto_bruto)
    print("TEXTO ORIGINAL:\n", texto_bruto)
    print("-" * 40)
    print("TEXTO ANONIMIZADO:\n", resultado)