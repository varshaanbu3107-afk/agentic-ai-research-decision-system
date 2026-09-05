from app.rag.vector_store import VectorStore


def main():

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL-ONLY TEST")
    print("=" * 60)

    query = "How can AI reduce software development costs?"

    print(f"\nQuery: {query}")

    print("\nLoading persistent vector store...")

    vector_store = VectorStore()
    vector_store.load()

    print("\nSearching vector store...")

    results = vector_store.search(
        query,
        top_k=5
    )

    print("\n" + "=" * 60)
    print("TOP RETRIEVED RESULTS")
    print("=" * 60)

    for number, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        print(f"\n--- RESULT {number} ---")

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"Source: "
            f"{document.metadata.get('source', 'Unknown')}"
        )

        print(
            f"Page: "
            f"{document.metadata.get('page', 'Unknown')}"
        )

        print("\nContent:")

        print(
            document.page_content[:1000]
        )

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()