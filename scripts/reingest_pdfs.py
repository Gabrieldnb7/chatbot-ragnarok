#!/usr/bin/env python3
"""
Script de ingestão limpa dos PDFs do PGD.

Limpa o ChromaDB existente e re-ingere todos os PDFs de data/pdfs/
em uma única passada, sem duplicatas.

Uso:
    conda run -n chatbot python scripts/reingest_pdfs.py
"""

import sys
import shutil
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone

# Adiciona src/ ao path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db" / "chroma_data"
COLLECTION_NAME = "ragnarok_knowledge_base"


def extract_pdf_text(raw_content: bytes) -> str:
    """Extrai texto de um PDF usando pymupdf, pypdf ou PyPDF2."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(stream=raw_content, filetype="pdf")
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if text:
                pages.append(f"\n\n[Página {i}]\n{text}")
        return "\n".join(pages)
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise RuntimeError("Instale pymupdf, pypdf ou PyPDF2 para ler PDFs.")

    reader = PdfReader(BytesIO(raw_content))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(f"\n\n[Página {i}]\n{text}")
    return "\n".join(pages)


def main():
    print("=" * 70)
    print("  REINGESTÃO LIMPA — PDFs do PGD")
    print("=" * 70)

    # ── Passo 1: Listar PDFs ──────────────────────────────────────────
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"\n  ❌ Nenhum PDF encontrado em {PDF_DIR}")
        return

    print(f"\n  📄 {len(pdf_files)} PDFs encontrados em {PDF_DIR}:")
    for f in pdf_files:
        size_kb = f.stat().st_size / 1024
        print(f"     • {f.name} ({size_kb:.0f} KB)")

    # ── Passo 2: Limpar ChromaDB existente ────────────────────────────
    print(f"\n  🗑️  Limpando ChromaDB em {VECTOR_DB_PATH}...")

    if VECTOR_DB_PATH.exists():
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )

        try:
            col = client.get_collection(name=COLLECTION_NAME)
            old_count = col.count()
            client.delete_collection(name=COLLECTION_NAME)
            print(f"     ✓ Collection '{COLLECTION_NAME}' removida ({old_count} chunks antigos)")
        except Exception:
            print(f"     ✓ Collection '{COLLECTION_NAME}' não existia")

        # Recriar collection limpa
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        import chromadb
        from chromadb.config import Settings

        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    print(f"     ✓ Collection limpa criada")

    # ── Passo 3: Pipeline de ingestão ─────────────────────────────────
    from ingestion import ingest_and_anonymize
    from chunking import chunk_document
    from embeddings import generate_embeddings

    total_chunks = 0
    resultados = []

    for pdf_file in pdf_files:
        print(f"\n  📖 Processando: {pdf_file.name}...")

        # 3a. Extrair texto do PDF
        raw_content = pdf_file.read_bytes()
        raw_text = extract_pdf_text(raw_content)
        if not raw_text.strip():
            print(f"     ⚠️  PDF vazio ou sem texto extraível")
            continue

        # 3b. Anonimizar
        cleaned_text = ingest_and_anonymize(raw_text)

        # 3c. Chunking semântico
        metadata = {
            "titulo": "Documento carregado",
            "fonte": pdf_file.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        chunks = chunk_document(cleaned_text, metadata)

        if not chunks:
            print(f"     ⚠️  Nenhum chunk gerado")
            continue

        # 3d. Gerar embeddings
        embedded_chunks = generate_embeddings(chunks)

        # 3e. Persistir no ChromaDB
        ids = [c["id"] for c in embedded_chunks]
        embeddings = [c["embedding"] for c in embedded_chunks]
        documents = [c["texto"] for c in embedded_chunks]
        metadatas = [c["metadata"] for c in embedded_chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        total_chunks += len(embedded_chunks)
        resultados.append((pdf_file.name, len(embedded_chunks)))
        print(f"     ✓ {len(embedded_chunks)} chunks indexados")

    # ── Passo 4: Resumo ──────────────────────────────────────────────
    print(f"\n  {'=' * 66}")
    print(f"  RESUMO DA INGESTÃO")
    print(f"  {'=' * 66}")
    print(f"\n  {'PDF':<40} {'Chunks':>8}")
    print(f"  {'─' * 48}")
    for nome, n in resultados:
        print(f"  {nome:<40} {n:>8}")
    print(f"  {'─' * 48}")
    print(f"  {'TOTAL':<40} {total_chunks:>8}")
    print(f"\n  ✅ Base limpa com {total_chunks} chunks únicos (collection: {COLLECTION_NAME})")
    print(f"  {'=' * 66}")


if __name__ == "__main__":
    main()
