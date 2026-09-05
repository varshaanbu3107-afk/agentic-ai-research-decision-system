import os
import pickle

import faiss
from sentence_transformers import SentenceTransformer

from app.rag.chunker import split_documents
from app.rag.loader import load_pdf


MODEL_NAME = "all-MiniLM-L6-v2"

VECTOR_STORE_DIR = "data/vector_store"

INDEX_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "index.faiss"
)

DOCUMENTS_FILE = os.path.join(
    VECTOR_STORE_DIR,
    "documents.pkl"
)


class VectorStore:
    """
    FAISS vector store with persistent storage.
    """

    def __init__(self, model_name=MODEL_NAME):

        print(
            f"Loading embedding model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        self.index = None
        self.documents = []

    def build(self, documents):
        """
        Create embeddings and build FAISS index.
        """

        if not documents:
            raise ValueError(
                "No documents provided."
            )

        texts = [
            document.page_content
            for document in documents
        ]

        print(
            f"Creating embeddings for {len(texts)} "
            "documents..."
        )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        self.documents = documents

    def save(self):
        """
        Save FAISS index and documents to disk.
        """

        if self.index is None:
            raise ValueError(
                "Cannot save an empty vector store."
            )

        os.makedirs(
            VECTOR_STORE_DIR,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            INDEX_FILE
        )

        with open(
            DOCUMENTS_FILE,
            "wb"
        ) as file:

            pickle.dump(
                self.documents,
                file
            )

        print(
            "\nVector store saved successfully."
        )

        print(
            f"Index: {INDEX_FILE}"
        )

        print(
            f"Documents: {DOCUMENTS_FILE}"
        )

    def load(self):
        """
        Load FAISS index and documents from disk.
        """

        if not os.path.exists(INDEX_FILE):
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_FILE}"
            )

        if not os.path.exists(DOCUMENTS_FILE):
            raise FileNotFoundError(
                f"Document file not found: "
                f"{DOCUMENTS_FILE}"
            )

        self.index = faiss.read_index(
            INDEX_FILE
        )

        with open(
            DOCUMENTS_FILE,
            "rb"
        ) as file:

            self.documents = pickle.load(
                file
            )

        print(
            "\nVector store loaded successfully."
        )

        print(
            f"Documents available: "
            f"{len(self.documents)}"
        )

    def search(
        self,
        query,
        top_k=3
    ):
        """
        Search the vector store using
        semantic similarity.
        """

        if self.index is None:
            raise ValueError(
                "Vector store has not been built "
                "or loaded."
            )

        if not self.documents:
            raise ValueError(
                "No documents are available."
            )

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        number_of_results = min(
            top_k,
            len(self.documents)
        )

        scores, indices = self.index.search(
            query_embedding,
            number_of_results
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append(
                {
                    "score": float(score),
                    "document": self.documents[index]
                }
            )

        return results


if __name__ == "__main__":

    pdf_path = (
        "data/documents/test_research.pdf"
    )

    print("\n" + "=" * 60)
    print("VECTOR STORE BUILD TEST")
    print("=" * 60)

    print("\nLoading PDF...")

    documents = load_pdf(
        pdf_path
    )

    print(
        f"Pages loaded: {len(documents)}"
    )

    print("\nSplitting document...")

    chunks = split_documents(
        documents
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    print("\nBuilding vector store...")

    vector_store = VectorStore()

    vector_store.build(
        chunks
    )

    vector_store.save()

    print("\n" + "=" * 60)
    print("PERSISTENCE TEST")
    print("=" * 60)

    print(
        "\nCreating a new VectorStore instance..."
    )

    loaded_store = VectorStore()

    loaded_store.load()

    query = (
        "How can AI improve customer "
        "support efficiency?"
    )

    print(
        f"\nSearching for: {query}"
    )

    results = loaded_store.search(
        query,
        top_k=3
    )

    print("\nSearch results:")

    for number, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        print(
            f"\n--- RESULT {number} ---"
        )

        print(
            f"Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('source')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page')}"
        )

        print(
            f"\n{document.page_content}"
        )