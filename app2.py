import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="C++ RAG ChatBot", layout="wide")
st.title("🗨️C++ RAG ChatBot")


@st.cache_resource
def load_vector_store(): 
    # step A: Load documents
    loader = TextLoader("C++_Introduction.txt", encoding="utf-8");
    documents = loader.load();

    # step B: Split text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20 
        # Chunk Overlap - 20 characters overlap
        # Overlap helps maintain context continuity
    );

    final_documents = text_splitter.split_documents(documents);
    # step C: Embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
        #This is the embedding model
    );
    # step D: Create FAISS Vector store
    # Converts each chunk to embedding, then stores them and makes searchable
    db = FAISS.from_documents(final_documents, embedding);

    # return faiss database
    return db


db = load_vector_store();

# llm = Ollama(model_name="gemma2:2b")
llm = OllamaLLM(model="gemma2:2b")

# Chat interface
text_input = st.text_input("Ask a question about c++")
if(text_input):
    with st.spinner(text="Thinking..."):
        docs = db.similarity_search(text_input);
        context = "\n".join([doc.page_content for doc in docs])
    prompt = f"""
    Answer the question using the context below
    
    Context: {context}
    Question: {text_input}
    Answer:
    """
    response = llm.invoke(prompt)
    st.subheader("Answers")
    st.write(response)
    