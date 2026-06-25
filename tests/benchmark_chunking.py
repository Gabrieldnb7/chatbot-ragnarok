#!/usr/bin/env python3
"""
Benchmark para o chunk_document() da Tarefa 02.

Métricas capturadas:
  - Número de chunks
  - Tamanho dos chunks (min, max, média, std)
  - Fronteiras: preview dos chunks gerados
  - Abreviações: análise detalhada de splits internos e externos
  - Ruído textual: artefatos como \n\n no meio de sentenças
  - Tempo de execução

Uso:
    python tests/benchmark_chunking.py             # roda e salva baseline
    python tests/benchmark_chunking.py --compare   # compara com baseline salvo
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunking import chunk_document, _SENTENCE_SEPARATOR_PATTERN

BENCHMARK_FILE = Path(__file__).parent / "benchmark_chunking_result.json"

# =====================================================================
# TEXTOS DE TESTE
# =====================================================================

TEXTO_CONTROLE = """
Procedimento de recuperação de acesso ao sistema institucional.
O usuário deve acessar a página oficial de autenticação e selecionar
a opção Esqueci minha senha. Em seguida, deve informar o e-mail
institucional cadastrado e aguardar o envio da mensagem com o link
temporário de redefinição.

O link deve ser utilizado dentro do prazo informado na mensagem.
Caso o e-mail não seja recebido, o usuário deve verificar as pastas
de spam e lixo eletrônico.

Depois da redefinição, a nova senha deve respeitar os critérios
exibidos na tela, incluindo tamanho mínimo e combinação de letras,
números e caracteres especiais.

Quando a conta estiver bloqueada por excesso de tentativas, o usuário
deve aguardar o período indicado antes de tentar novamente. Se o
bloqueio permanecer, o chamado deve ser encaminhado ao atendimento
responsável por acessos.
""".strip()

TEXTO_ABREVIACOES = """
O Dr. Carlos Eduardo da Silva compareceu à reunião com a Sra. Maria
Aparecida para discutir o contrato. O Sr. João Pedro, representante
da empresa, apresentou os documentos necessários.

O Dr. Silva solicitou a análise do processo etc. O prazo estipulado
pela Sra. Maria foi de 30 dias corridos. Após esse período, o
contrato será automaticamente renovado.

A Sra. Aparecida confirmou que a empresa X Ltda. está apta a
participar da licitação. O Dr. Carlos fará a auditoria final.
""".strip()

TEXTO_LONGO = """
Seção 1: Introdução ao sistema de atendimento institucional.

O sistema de atendimento institucional foi desenvolvido para
centralizar as solicitações dos usuários e permitir o rastreamento
completo do ciclo de vida de cada chamado. Este documento descreve
os procedimentos operacionais padrão que devem ser seguidos por
todos os atendentes.

Seção 2: Abertura de chamados.

Para abrir um novo chamado, o atendente deve acessar o módulo de
cadastro e preencher os campos obrigatórios: nome do solicitante,
setor, descrição do problema e nível de urgência. O sistema
automaticamente atribuirá um número de protocolo.

A descrição do problema deve ser clara e objetiva. Evite termos
técnicos desnecessários. Inclua o passo a passo realizado até o
momento do erro. Informe também qual sistema estava sendo utilizado
e a mensagem de erro exibida.

Seção 3: Classificação e priorização.

Cada chamado recebe uma classificação baseada no tipo de solicitação:
incidente, requisição de serviço ou acesso à informação. A prioridade
é definida automaticamente pelo sistema com base na combinação de
urgência e impacto.

Chamados classificados como críticos devem ser atendidos em até
2 horas úteis. Chamados de alta prioridade têm prazo de 8 horas
úteis. Chamados de média prioridade têm prazo de 24 horas úteis.
Chamados de baixa prioridade têm prazo de 72 horas úteis.

Seção 4: Procedimentos de escalonamento.

Quando o atendente não conseguir resolver o chamado dentro do prazo
estabelecido, o caso deve ser escalonado para o nível 2 de suporte.
O escalonamento deve incluir um resumo do que já foi testado e o
motivo pelo qual o chamado não pôde ser resolvido no nível 1.

O nível 2 de suporte tem acesso a ferramentas administrativas e
pode realizar alterações na configuração dos sistemas. Caso o
chamado ainda não seja resolvido, ele é escalonado para o nível 3,
que corresponde à equipe de engenharia responsável pelo sistema.

Seção 5: Encerramento de chamados.

O chamado pode ser encerrado quando o usuário confirmar que o problema
foi resolvido. O atendente deve registrar a solução aplicada e o
tempo total de atendimento. Chamados encerrados sem confirmação do
usuário devem ser reabertos automaticamente.

Chamados que permanecerem sem atividade por mais de 15 dias úteis
são automaticamente suspensos. O usuário é notificado por e-mail
antes da suspensão e pode reativar o chamado dentro de 5 dias úteis.
""".strip()


# =====================================================================
# MÉTRICAS
# =====================================================================

# Lista de abreviações portuguesas que NÃO devem ser confundidas com fim de frase
_ABREVIACOES_CONHECIDAS = [
    "Dr", "Sr", "Sra", "Srta", "Srtas", "Srs", "Sras",
    "etc", "Ltda",
    "obs", "art", "pag", "vol", "cap", "ed",
    "Ex", "V",  # Excelentíssimo, Vossa
]
_PAT_ABREV_FIM = re.compile(
    r"\b(" + "|".join(_ABREVIACOES_CONHECIDAS) + r")\.(?=\s)",
    re.UNICODE,
)

# Padrão para detectar abreviações que antecedem \n\n dentro de chunks
# Procura por abreviatura + "." + imediatamente seguido de \n\n (sem espaço entre)
_PAT_ABREV_NN = re.compile(
    r"\b(" + "|".join(_ABREVIACOES_CONHECIDAS) + r")\.\n\n",
    re.UNICODE,
)


def analisar_abreviacoes(texto_original: str, chunks: list) -> dict:
    """
    Analisa como as abreviações foram tratadas no chunking.

    1. Conta abreviações no texto original
    2. Verifica se dentro dos chunks alguma abreviatura foi seguida
       de \n\n (artefato do split incorreto)
    3. Verifica se alguma abreviatura ficou como última palavra de um chunk
    """
    # Encontra todas as abreviações no texto original
    ocorrencias = list(_PAT_ABREV_FIM.finditer(texto_original))
    total_abrev = len(ocorrencias)

    # Procura por "Abrev.\n\n" dentro de cada chunk
    abrev_com_artefato = 0
    abrev_quebrada = 0
    artefatos_detectados = []

    for chunk in chunks:
        texto = chunk["texto"]

        # Método 1: procura diretamente por "Abrev.\n\n" no chunk
        for m in _PAT_ABREV_NN.finditer(texto):
            abrev_com_artefato += 1
            ctx = texto[max(0, m.start() - 10):m.end() + 15].replace("\n", "\\n")
            grp = m.group().replace(chr(10), "\\n")
            artefatos_detectados.append(f"'{grp}' em '{ctx}'")

        # Método 2: verifica se chunk termina com abreviatura
        texto_clean = texto.rstrip()
        for m in _PAT_ABREV_FIM.finditer(texto_clean):
            if m.end() == len(texto_clean):
                abrev_quebrada += 1

    return {
        "total_abreviacoes": total_abrev,
        "com_artefato_nn": abrev_com_artefato,
        "no_fim_do_chunk": abrev_quebrada,
        "integridade_pct": round(
            (1 - (abrev_com_artefato + abrev_quebrada) / total_abrev) * 100, 1
        ) if total_abrev > 0 else 100.0,
        "artefatos": artefatos_detectados[:10],
    }


def medir_tamanhos_chunks(chunks: list) -> dict:
    tamanhos = [len(c["texto"]) for c in chunks]
    if not tamanhos:
        return {"min": 0, "max": 0, "media": 0, "std": 0, "total": 0}

    media = sum(tamanhos) / len(tamanhos)
    var = sum((t - media) ** 2 for t in tamanhos) / len(tamanhos)
    return {
        "min": min(tamanhos),
        "max": max(tamanhos),
        "media": round(media, 1),
        "std": round(var ** 0.5, 1),
        "total": len(chunks),
    }


def medir_ruido_sentenca(chunks: list) -> dict:
    """
    Detecta \n\n no meio de sentenças (artefato de split incorreto).

    Um \n\n é considerado ruído quando:
    - Não está após pontuação de fim de frase VÁLIDA
    - OU está após uma abreviatura conhecida (Dr., Sr., etc.)
    """
    total_nn_interno = 0
    locais = []

    # Pattern para detectar abreviação antes de \n\n
    abrev_antes_nn = re.compile(
        r"\b(" + "|".join(_ABREVIACOES_CONHECIDAS) + r")\.(?=\s*\n\n)",
        re.UNICODE,
    )

    for chunk in chunks:
        texto = chunk["texto"]
        for m in re.finditer(r"\n\n", texto):
            pos = m.start()
            antes = texto[pos - 1] if pos > 0 else " "

            # Se antes do \n\n tem uma abreviatura → ruído
            # Verifica olhando pra trás
            trecho_anterior = texto[max(0, pos - 10):pos]
            eh_abrev = bool(abrev_antes_nn.search(trecho_anterior + "\n\n"))

            if eh_abrev:
                total_nn_interno += 1
            elif antes not in (".", "!", "?", ":", ";"):
                total_nn_interno += 1
            else:
                continue  # \n\n em fim de frase legítimo

            ctx = texto[max(0, pos - 15):pos + 15].replace("\n", "\\n")
            locais.append(ctx[:50])

    return {
        "total_nn_interno": total_nn_interno,
        "locais": locais[:8],
    }


# =====================================================================
# MAIN
# =====================================================================

METADATA_PADRAO = {
    "titulo": "Benchmark - Procedimentos Operacionais",
    "fonte": "Documento de teste - benchmark interno",
}

CENARIOS = [
    ("controle", TEXTO_CONTROLE, "Texto controle — sem abreviações complicadas"),
    ("abreviacoes", TEXTO_ABREVIACOES, "Texto rico em abreviações (Dr., Sr., Sra., etc.)"),
    ("longo", TEXTO_LONGO, "Texto longo — múltiplas seções e parágrafos"),
]


def benchmark_texto(nome: str, texto: str, metadata: dict) -> dict:
    t0 = time.perf_counter()
    chunks = chunk_document(texto, metadata)
    elapsed = time.perf_counter() - t0

    tamanhos = medir_tamanhos_chunks(chunks)
    abreviacoes = analisar_abreviacoes(texto, chunks)
    ruido = medir_ruido_sentenca(chunks)

    fronteiras = []
    for c in chunks:
        preview = c["texto"][:80].replace("\n", " ").strip()
        fronteiras.append(f"Chunk {c['id']}: \"{preview}\"")

    # Detalhe dos splits internos (mostra a posição de cada \n\n)
    splits_internos = []
    for c in chunks:
        for m in re.finditer(r"\n\n", c["texto"]):
            ctx = c["texto"][max(0, m.start() - 20):m.end() + 20]
            splits_internos.append(ctx.replace("\n", "\\n").strip())

    return {
        "nome": nome,
        "texto_len": len(texto),
        "tempo_seg": round(elapsed, 4),
        "chunks": tamanhos,
        "abreviacoes": abreviacoes,
        "ruido": ruido,
        "fronteiras": fronteiras,
        "splits_internos": splits_internos[:10],
    }


def rodar_benchmark() -> dict:
    print("=" * 70)
    print("  BENCHMARK — chunk_document() da Tarefa 02")
    print("=" * 70)
    print()

    resultados = {}
    for nome, texto, desc in CENARIOS:
        print(f"  [{nome}] {desc}")
        print(f"         {len(texto)} caracteres")
        r = benchmark_texto(nome, texto, METADATA_PADRAO)
        resultados[nome] = r
        print(f"         → {r['chunks']['total']} chunks em {r['tempo_seg']:.4f}s")
        a = r["abreviacoes"]
        print(f"         → abreviações: {a['total_abreviacoes']} total, "
              f"{a['com_artefato_nn']} com \\n\\n, {a['no_fim_do_chunk']} no fim")
        if a["artefatos"]:
            for art in a["artefatos"][:3]:
                print(f"           • {art}")
        ruido = r["ruido"]
        if ruido["total_nn_interno"]:
            print(f"         → ⚠️ {ruido['total_nn_interno']} quebras \\n\\n no meio de sentenças")
            for loc in ruido["locais"][:3]:
                print(f"           • {loc}")
        if r["splits_internos"]:
            print(f"         → splits \\n\\n detectados no chunk:")
            for s in r["splits_internos"][:3]:
                print(f"           • ...{s}...")
        print()

    total_chunks = sum(r["chunks"]["total"] for r in resultados.values())
    total_artefatos = sum(r["abreviacoes"]["com_artefato_nn"] for r in resultados.values())
    total_quebradas = sum(r["abreviacoes"]["no_fim_do_chunk"] for r in resultados.values())
    total_abrev = sum(r["abreviacoes"]["total_abreviacoes"] for r in resultados.values())
    total_ruido = sum(r["ruido"]["total_nn_interno"] for r in resultados.values())
    tempo_total = sum(r["tempo_seg"] for r in resultados.values())

    resumo = {
        "versao": "baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cenarios": resultados,
        "agregado": {
            "total_chunks": total_chunks,
            "abreviacoes_total": total_abrev,
            "abreviacoes_com_artefato": total_artefatos,
            "abreviacoes_quebradas": total_quebradas,
            "abreviacoes_integridade_pct": round(
                (1 - (total_artefatos + total_quebradas) / total_abrev) * 100, 1
            ) if total_abrev > 0 else 100.0,
            "ruido_nn_interno": total_ruido,
            "tempo_total_seg": round(tempo_total, 4),
        }
    }

    print("-" * 70)
    print("  RESUMO AGREGADO")
    print(f"    Total de chunks:               {resumo['agregado']['total_chunks']}")
    print(f"    Abreviações com artefato:       {resumo['agregado']['abreviacoes_com_artefato']}/{resumo['agregado']['abreviacoes_total']}")
    print(f"    Abreviações no fim do chunk:    {resumo['agregado']['abreviacoes_quebradas']}/{resumo['agregado']['abreviacoes_total']}")
    print(f"    Integridade abreviações:        {resumo['agregado']['abreviacoes_integridade_pct']}%")
    print(f"    Ruído \\n\\n em sentenças:         {resumo['agregado']['ruido_nn_interno']}")
    print(f"    Tempo total:                    {resumo['agregado']['tempo_total_seg']}s")
    print("=" * 70)

    BENCHMARK_FILE.write_text(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"\n  Resultados salvos em: {BENCHMARK_FILE}")
    return resumo


def comparar_com_baseline():
    if not BENCHMARK_FILE.exists():
        print(f"  Arquivo de baseline não encontrado: {BENCHMARK_FILE}")
        print("  Rode primeiro sem --compare para gerar a baseline.")
        return

    baseline = json.loads(BENCHMARK_FILE.read_text())
    print("=" * 70)
    print("  COMPARAÇÃO — baseline vs versão atual")
    print(f"  Baseline: {baseline['timestamp']} (versao: {baseline['versao']})")
    print("=" * 70)

    resultados_atuais = {}
    for nome, texto, desc in CENARIOS:
        resultados_atuais[nome] = benchmark_texto(nome, texto, METADATA_PADRAO)

    # Salva resultado atual para comparação
    atual = {
        "versao": "atual",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cenarios": resultados_atuais,
    }

    print(f"\n{'':>30} {'Baseline':>12} {'Agora':>12} {'Δ':>10}")
    print("-" * 66)

    for nome, _, _ in CENARIOS:
        bl = baseline["cenarios"][nome]
        at = resultados_atuais[nome]
        print(f"  [{nome}]")

        for rotulo, chave in [
            ("Chunks", "chunks.total"),
            ("Tempo (s)", "tempo_seg"),
            ("Abrev. c/ artefato", "abreviacoes.com_artefato_nn"),
            ("Abrev. quebradas", "abreviacoes.no_fim_do_chunk"),
            ("Ruido \\n\\n sentenca", "ruido.total_nn_interno"),
        ]:
            v_bl = bl
            v_at = at
            for k in chave.split("."):
                v_bl = v_bl[k]
                v_at = v_at[k]
            if isinstance(v_bl, float):
                print(f"  {rotulo:>30} {v_bl:>12.4f} {v_at:>12.4f} {v_at - v_bl:>+10.4f}")
            else:
                print(f"  {rotulo:>30} {v_bl:>12} {v_at:>12} {v_at - v_bl:>+10}")

        print()

    return atual


if __name__ == "__main__":
    if "--compare" in sys.argv:
        comparar_com_baseline()
    else:
        rodar_benchmark()
