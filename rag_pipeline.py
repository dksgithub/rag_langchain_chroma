from langchain_chroma import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

def build_retriever():
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(collection_name="rag_collection", embedding_function=embedding_function)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    return retriever

def build_qa_chain():
    retriever = build_retriever()
    llm = OpenAI(temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
    return chain
