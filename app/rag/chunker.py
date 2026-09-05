from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150
):
    """
    Split documents into smaller chunks for RAG processing.
    """

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents(documents)

    return chunks


if __name__ == "__main__":

    from app.rag.loader import load_pdf

    pdf_path = "data/documents/test_research.pdf"

    documents = load_pdf(pdf_path)

    chunks = split_documents(documents)

    print("\n" + "=" * 60)
    print("DOCUMENT CHUNKER TEST")
    print("=" * 60)

    print(f"\nOriginal documents: {len(documents)}")
    print(f"Number of chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks, start=1):

        print(f"\n--- CHUNK {index} ---")
        print(chunk.page_content)

        print(f"\nMetadata: {chunk.metadata}")