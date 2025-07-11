from documents import documents
from embedding import embed_texts
from db import get_chroma_collection, store_documents
from rag_pipeline import build_qa_chain

def main():
    print("Embedding documents...")
    embeddings = embed_texts(documents)

    print("Storing in ChromaDB...")
    collection = get_chroma_collection()
    store_documents(collection, documents, embeddings)

    print("Building QA Chain...")
    qa_chain = build_qa_chain()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        result = qa_chain({"query": query})
        print("\nAnswer:", result["result"])
        print("Top Docs:")
        for doc in result["source_documents"]:
            print("-", doc.page_content)

if __name__ == "__main__":
    main()
