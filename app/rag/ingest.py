from pathlib import Path

from app.rag.loader import load_pdf
from app.rag.chunker import split_documents
from app.rag.vector_store import VectorStore


DOCUMENTS_DIR = Path("data/documents/research")


def find_pdf_files():
    """
    Find all PDF files inside the documents directory.
    """

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    pdf_files = list(
    DOCUMENTS_DIR.rglob("*.pdf")
)
    

    return sorted(pdf_files)


def ingest_documents():

    print("\n" + "=" * 60)
    print("DOCUMENT INGESTION PIPELINE")
    print("=" * 60)

    pdf_files = find_pdf_files()

    if not pdf_files:
        raise ValueError(
            "No PDF files found in data/documents/"
        )

    print(
        f"\nPDF files discovered: {len(pdf_files)}"
    )

    all_documents = []

    for pdf_path in pdf_files:

        print("\n" + "-" * 60)
        print(f"Processing: {pdf_path.name}")
        print("-" * 60)

        documents = load_pdf(
            str(pdf_path)
        )

        print(
            f"Pages loaded: {len(documents)}"
        )

        chunks = split_documents(
            documents
        )

        print(
            f"Chunks created: {len(chunks)}"
        )

        all_documents.extend(
            chunks
        )

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)

    print(
        f"\nTotal PDFs: {len(pdf_files)}"
    )

    print(
        f"Total chunks: {len(all_documents)}"
    )

    print("\nBuilding unified vector store...")

    vector_store = VectorStore()

    vector_store.build(
        all_documents
    )

    vector_store.save()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    ingest_documents()