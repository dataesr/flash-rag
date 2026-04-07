from chromadb.api import ClientAPI
from chromadb import (
    PersistentClient,
    Schema,
    VectorIndexConfig,
    Collection,
)

# from chromadb.utils.embedding_functions import ChromaBm25EmbeddingFunction
from src.mistral import MistralEmbeddingFunction

DB_DIR = "./db"
COLLECTION_NAME = "flash-notes"


def get_client() -> ClientAPI:
    print(f"[chromadb] Initializing ChromaDB at {DB_DIR}")
    client = PersistentClient(path=DB_DIR)
    return client


client = get_client()


def build_schema():
    """Build a ChromaDB schema with a BM25 sparse index on document text.

    The dense index (Mistral embeddings) is set via `embedding_function`
    on the collection. The sparse index is added here for keyword/BM25
    retrieval, enabling hybrid search via RRF.
    """
    schema = Schema()
    schema.create_index(
        config=VectorIndexConfig(
            space="cosine",
            embedding_function=MistralEmbeddingFunction(),
        ),
    )
    # schema.create_index(
    #     key="sparse_embedding",
    #     config=SparseVectorIndexConfig(
    #         source_key=str(K.DOCUMENT),
    #         embedding_function=ChromaBm25EmbeddingFunction(), # not implemented yet for local dev
    #     ),
    # )

    return schema


def get_collection(reset: bool = False) -> Collection:
    if reset:
        print(f"[chromadb] Resetting collection '{COLLECTION_NAME}'")
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        schema=build_schema(),
    )
    print(f"[chromadb] Collection '{COLLECTION_NAME}' ready (count={collection.count()})")
    return collection
