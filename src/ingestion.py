import re
import spacy

# 🟡 4. CORREÇÃO: Variável global para lazy loading do spaCy
_nlp = None

def carregar_spacy():
    """Carrega o modelo do spaCy apenas quando necessário (Lazy Loading)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("pt_core_news_sm")
        except OSError:
            raise OSError("Modelo não encontrado. Execute: python -m spacy download pt_core_news_sm")
    return _nlp

def is_valid_luhn(n: str) -> bool:
    """
    🔴 2. CORREÇÃO: Verifica se uma string numérica passa no algoritmo de Luhn.
    Garante que o número não é um protocolo ou ID qualquer, mas um cartão válido.
    """
    # Deixa apenas os números
    n = ''.join(filter(str.isdigit, n))
    if not n or len(n) < 13: 
        return False
    
    soma = 0
    alt = False
    for d in reversed(n):
        d = int(d)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        soma += d
        alt = not alt
    return soma % 10 == 0

def substituir_cartao(match) -> str:
    """Função auxiliar para o regex substituir apenas cartões reais."""
    texto_capturado = match.group(0)
    if is_valid_luhn(texto_capturado):
        return '[DADO BANCARIO REMOVIDO]'
    return texto_capturado

def ingest_and_anonymize(file_content: str) -> str:
    """
    Ingere um texto bruto, realiza limpeza estrutural e anonimiza dados sensíveis.
    """
    # 🟡 3. CORREÇÃO: Erro silencioso substituído por TypeError explícito
    if not isinstance(file_content, str):
        raise TypeError(f"Esperado string em file_content, recebido {type(file_content).__name__}")

    # =================================================================
    # ETAPA 1: Limpeza Estrutural Básica
    # =================================================================
    texto_limpo = re.sub(r'\s+', ' ', file_content).strip()

    # =================================================================
    # ETAPA 2: Anonimização Determinística (Expressões Regulares)
    # =================================================================
    
    # 🟡 5. CORREÇÃO: Regex para CPF aceitando formatos com ou sem pontuação explícita
    texto_limpo = re.sub(r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b', '[CPF REMOVIDO]', texto_limpo)
    
    # Padrão para E-mails
    texto_limpo = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REMOVIDO]', texto_limpo)
    
    # 🔴 1. CORREÇÃO: Telefone agora EXIGE DDD (ex: 11) para evitar que protocolos sejam capturados
    texto_limpo = re.sub(r'\b(?:\+?55\s?)?\(?\d{2}\)?\s?(?:9\s?)?\d{4}[-\s]?\d{4}\b', '[TELEFONE REMOVIDO]', texto_limpo)
    
    # 🔴 2. CORREÇÃO: Integração do algoritmo de Luhn ao Regex de cartão de crédito
    padrao_cartao = r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
    texto_limpo = re.sub(padrao_cartao, substituir_cartao, texto_limpo)

    # =================================================================
    # ETAPA 3: Anonimização Contextual (spaCy - NER)
    # =================================================================
    
    # Carrega o spaCy apenas agora (se nunca rodar a função, não gasta memória atoa)
    nlp = carregar_spacy()
    doc = nlp(texto_limpo)
    texto_anonimizado = texto_limpo
    
    for ent in reversed(doc.ents):
        # ℹ️ 6. CORREÇÃO: Adicionado 'ORG' (Organizações) além de 'PER' (Pessoas)
        if ent.label_ in ["PER", "ORG"]:
            inicio = ent.start_char
            fim = ent.end_char
            tipo = "NOME" if ent.label_ == "PER" else "ORGANIZAÇÃO"
            texto_anonimizado = texto_anonimizado[:inicio] + f"[{tipo} REMOVIDO]" + texto_anonimizado[fim:]

    return texto_anonimizado

# =================================================================
# TESTES DE MESA (Provando que as correções funcionam)
# =================================================================
if __name__ == "__main__":
    texto_bruto = """
    Documento de Teste de Falsos Positivos.
    Protocolo de atendimento: 1234-5678 (NÃO DEVE SUMIR)
    Data do sistema: 2025-1234 (NÃO DEVE SUMIR)
    ID do processo: 4321 8765 2109 6543 (NÃO DEVE SUMIR)
    
    Dados Reais:
    Ligar para Maria da Silva no (11) 98765-4321 ou email maria@teste.com.
    CPF sem ponto: 12345678900.
    A empresa Nubank processou o cartão 4532 1122 3344 5566.
    """
    
    try:
        resultado = ingest_and_anonymize(texto_bruto)
        print("TEXTO ANONIMIZADO:\n", resultado)
        
        # Testando o TypeError
        ingest_and_anonymize(123)
    except TypeError as e:
        print("\n[SUCESSO] Exceção capturada com sucesso:", e)