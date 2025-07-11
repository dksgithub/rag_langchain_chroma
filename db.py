import chromadb
from chromadb.utils import embedding_functions

def get_chroma_collection():
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="rag_collection")
    return collection

def store_documents(collection, documents, embeddings):
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{{i}}" for i in range(len(documents))]
    )
