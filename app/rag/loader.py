from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document


def load_pdf(file_path: str):
    """
    Load a PDF and return its pages as LangChain Documents.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    if path.suffix.lower() != ".pdf":
        raise ValueError(
            "The provided file must be a PDF."
        )

    reader = PdfReader(str(path))

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        document = Document(
            page_content=text,
            metadata={
                "source": str(path),
                "page": page_number
            }
        )

        documents.append(document)

    return documents


if __name__ == "__main__":

    pdf_path = "data/documents/test_research.pdf"

    documents = load_pdf(pdf_path)

    print("\n" + "=" * 60)
    print("PDF LOADER TEST")
    print("=" * 60)

    print(f"\nNumber of pages: {len(documents)}")

    for document in documents:
        print("\n--- PAGE ---")
        print(document.page_content)
        print(f"\nMetadata: {document.metadata}")