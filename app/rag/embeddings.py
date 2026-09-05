from langchain_huggingface import HuggingFaceEmbeddings


def create_embeddings():
    """
    Create and return the embedding model.
    """

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("EMBEDDING MODEL TEST")
    print("=" * 60)

    embeddings = create_embeddings()

    test_text = "AI agents can automate routine customer support tasks."

    vector = embeddings.embed_query(test_text)

    print(f"\nEmbedding dimension: {len(vector)}")
    print(f"First 10 values: {vector[:10]}")