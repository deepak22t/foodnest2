# tools/rag.py

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
import weaviate
from weaviate.classes import init as weaviate_init
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter
from langchain_weaviate.vectorstores import WeaviateVectorStore

load_dotenv()

RAG_INDEX_NAME = "RAGIndex"

# =========================
# CONFIG
# =========================
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
# gRPC init check defaults to ~2s in the client; slow/Wi‑Fi networks often need more
WEAVIATE_INIT_TIMEOUT = float(os.getenv("WEAVIATE_INIT_TIMEOUT", "30"))
WEAVIATE_QUERY_TIMEOUT = float(os.getenv("WEAVIATE_QUERY_TIMEOUT", "30"))
WEAVIATE_INSERT_TIMEOUT = float(os.getenv("WEAVIATE_INSERT_TIMEOUT", "90"))
# Last resort if gRPC health check to WCD is blocked: set to true (not recommended for prod)
WEAVIATE_SKIP_INIT_CHECKS = os.getenv("WEAVIATE_SKIP_INIT_CHECKS", "").lower() in (
    "1",
    "true",
    "yes",
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# INIT MODELS
# =========================
def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        api_key=GROQ_API_KEY
    )

def get_embeddings():
    return OpenAIEmbeddings()  # can replace later

def get_weaviate_client():
    """Weaviate Python client v4 (WeaviateClient), required by weaviate-client>=4."""
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        raise ValueError("WEAVIATE_URL and WEAVIATE_API_KEY must be set in the environment")
    extra = weaviate_init.AdditionalConfig(
        timeout=weaviate_init.Timeout(
            init=WEAVIATE_INIT_TIMEOUT,
            query=WEAVIATE_QUERY_TIMEOUT,
            insert=WEAVIATE_INSERT_TIMEOUT,
        )
    )
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=WEAVIATE_API_KEY,
        additional_config=extra,
        skip_init_checks=WEAVIATE_SKIP_INIT_CHECKS,
    )


def ensure_rag_index_schema(client) -> None:
    """
    Ensure RAGIndex has text, user_id, and source properties.
    Adds missing properties to existing collections. New installs get all three.
    Re-upload documents after this change; older chunks may lack user_id and will not match filters.
    """
    if not client.collections.exists(RAG_INDEX_NAME):
        client.collections.create(
            name=RAG_INDEX_NAME,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="user_id", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
        )
        return

    col = client.collections.get(RAG_INDEX_NAME)
    for prop_name in ("user_id", "source"):
        try:
            col.config.add_property(Property(name=prop_name, data_type=DataType.TEXT))
        except Exception:
            pass


# =========================
# CHUNKING (SMART)
# =========================
def chunk_text(text, source_name="uploaded_file", user_id: str = "default"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    docs = [
        Document(
            page_content=text,
            metadata={"source": source_name, "user_id": user_id},
        )
    ]
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        title = chunk.metadata.get("source", "Unknown")
        chunk.page_content = f"[Title: {title}]\n{chunk.page_content}"

    return chunks

# =========================
# STORE IN WEAVIATE
# =========================
def store_chunks(chunks):
    client = get_weaviate_client()
    try:
        ensure_rag_index_schema(client)
        embeddings = get_embeddings()
        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=RAG_INDEX_NAME,
            text_key="text",
            embedding=embeddings,
            attributes=["user_id", "source"],
        )
        vectorstore.add_documents(chunks)
        return vectorstore
    finally:
        try:
            client.close()
        except Exception:
            pass

# =========================
# RETRIEVE + RERANK
# =========================
def simple_rerank(query, docs):
    scored = []
    for doc in docs:
        score = sum(word in doc.page_content.lower() for word in query.lower().split())
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:3]]

# =========================
# RAG PIPELINE
# =========================
class RAGPipeline:
    def __init__(self):
        self.client = get_weaviate_client()
        ensure_rag_index_schema(self.client)
        self.embeddings = get_embeddings()
        self.llm = get_llm()

        self.vectorstore = WeaviateVectorStore(
            client=self.client,
            index_name=RAG_INDEX_NAME,
            text_key="text",
            embedding=self.embeddings,
            attributes=["user_id", "source"],
        )

    def add_document(self, text, filename, user_id: str = "default"):
        chunks = chunk_text(text, filename, user_id=user_id)
        self.vectorstore.add_documents(chunks)

    def query(self, user_message: str, user_id: str = "default"):
        user_filter = Filter.by_property("user_id").equal(user_id)
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3, "filters": user_filter},
        )

        docs = retriever.invoke(user_message)

        top_docs = simple_rerank(user_message, docs)

        context = "\n\n".join([doc.page_content for doc in top_docs])

        prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question: {user_message}
"""

        response = self.llm.invoke(prompt)

        return response.content
