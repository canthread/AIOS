"""
embedder.py
-----------
Ingestion layer — embedding pipeline.

Takes documents produced by docker_crawler.py, wraps them as LlamaIndex
Document objects, embeds them using Ollama (nomic-embed-text), and stores
them in ChromaDB.

OS framing: this is the memory subsystem. The crawler reads hardware state,
the embedder writes it to persistent addressable memory (ChromaDB). The
retriever will read it back like a filesystem lookup.
"""

import chromadb
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

from crawler import DockerCrawler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "homelab_state"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Build LlamaIndex documents from crawler output
# ---------------------------------------------------------------------------

def build_documents(crawler_results: dict) -> list[Document]:
    """
    Convert crawler dataclass objects into LlamaIndex Document objects.

    Each Document has:
      - text:     the plain text that gets embedded (from to_text())
      - metadata: structured fields stored alongside the vector for filtering

    Metadata is not embedded — it's stored as-is and returned with the chunk
    at retrieval time. Use it for filtering (e.g. "only containers on the
    mediastack network") or for giving the LLM extra context alongside the
    retrieved text.
    """
    documents = []

    for container in crawler_results["containers"]:
        documents.append(Document(
            text=container.to_text(),
            metadata={
                "type": "container",
                "name": container.name,
                "status": container.status,
                "compose_project": container.compose_project or "",
                "compose_service": container.compose_service or "",
                "image": container.image,
            }
        ))

    for network in crawler_results["networks"]:
        documents.append(Document(
            text=network.to_text(),
            metadata={
                "type": "network",
                "name": network.name,
                "driver": network.driver,
            }
        ))

    for volume in crawler_results["volumes"]:
        documents.append(Document(
            text=volume.to_text(),
            metadata={
                "type": "volume",
                "name": volume.name,
                "mountpoint": volume.mountpoint,
            }
        ))

    return documents


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class Embedder:
    """
    Connects to ChromaDB and embeds documents via Ollama.

    OS framing: this is the memory manager. It takes structured data from
    the crawler (hardware state) and writes it into addressable persistent
    storage (ChromaDB). The retriever reads it back.
    """

    def __init__(self):
        # Connect to ChromaDB
        self.chroma_client = chromadb.HttpClient(
            host=CHROMA_HOST,
            port=CHROMA_PORT,
        )

        # Get or create the collection
        # A collection is a named namespace in ChromaDB — all homelab state
        # lives in one collection, separated by metadata type field
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
        )

        # Ollama embedding model
        self.embed_model = OllamaEmbedding(
            model_name=EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # LlamaIndex vector store wrapper around the ChromaDB collection
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)

        # StorageContext tells LlamaIndex where to persist data
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )

    def embed(self, documents: list[Document]) -> VectorStoreIndex:
        """
        Embed a list of LlamaIndex Documents and store them in ChromaDB.

        VectorStoreIndex.from_documents() handles:
          1. Chunking each document (splitting long text into overlapping chunks)
          2. Calling the embedding model on each chunk
          3. Writing the vectors + original text + metadata to ChromaDB

        Returns the index object, which you can use immediately for retrieval.
        """
        print(f"[embedder] Embedding {len(documents)} documents...")

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            show_progress=True,
        )

        print(f"[embedder] Done. Collection '{COLLECTION_NAME}' now has "
              f"{self.collection.count()} vectors.")

        return index

    def load_index(self) -> VectorStoreIndex:
        """
        Load an existing index from ChromaDB without re-embedding.
        Use this after the first run — no need to re-embed if the state
        hasn't changed.
        """
        return VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
        )

    def wipe_collection(self):
        """
        Delete and recreate the collection. Use when you want a clean
        re-ingestion of current stack state (e.g. after major changes).
        """
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        print(f"[embedder] Collection '{COLLECTION_NAME}' wiped and recreated.")


# ---------------------------------------------------------------------------
# Retrieval test
# ---------------------------------------------------------------------------

def retrieval_test(index: VectorStoreIndex, embed_model: OllamaEmbedding):
    """
    Sanity check: run a few queries against the index and print results.
    This is the validation step — if these return sensible answers, the
    memory layer is working. If not, fix retrieval before touching the
    planning layer.
    """
    test_queries = [
        "what port is Radarr on",
        "which containers are on the mediastack network",
        "what volumes does Jellyfin use",
        "what is the subnet of the mediastack network",
        ]
    # as_retriever does pure vector similarity — no LLM involved.
    # as_query_engine synthesizes an answer via LLM (defaults to OpenAI).
    retriever = index.as_retriever(
            embed_model=embed_model,
            similarity_top_k=3,
            )

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        response = retriever.retrieve(query)
        for i, node in enumerate(response):
            print(f"Result {i+1} (score: {node.score:.4f}):")
            print(node.text)
            print(f"Metadata: {node.metadata}")
            print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # 1. Crawl current Docker state
    print("[crawler] Reading Docker socket...")
    crawler = DockerCrawler()
    try:
        results = crawler.crawl_all()
        print(f"[crawler] Found {len(results['containers'])} containers, "
              f"{len(results['networks'])} networks, "
              f"{len(results['volumes'])} volumes.")
    finally:
        crawler.close()

    # 2. Build LlamaIndex documents
    documents = build_documents(results)
    print(f"[embedder] Built {len(documents)} documents.")

    # 3. Embed into ChromaDB
    embedder = Embedder()

    # Wipe first if you want a clean re-ingestion
    # embedder.wipe_collection()

    index = embedder.embed(documents)

    # 4. Run retrieval test
    retrieval_test(index, embedder.embed_model)
