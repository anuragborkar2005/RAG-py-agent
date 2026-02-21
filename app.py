import os
import streamlit as st
from dotenv import load_dotenv

#Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Windows + .
# Step 1:  Page Configuration
st.set_page_config(page_title="C++ RAG Chatbot", page_icon="🗨️" )
st.title("🗨️ C++ RAG Chatbot")
st.write("Ask any question related to c++ introduction")

# Step 2: Load Environment variables
load_dotenv()

# Step 3: Cache document loading
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

# Vector database runs only once because of cache concepts
db = load_vector_store();

# User input
query = st.text_input("Enter your question about c++ :");

if(query):
    # Converts user questions to embeddings
    # Searches FAISS database
    # Returns top 3 similar chunks
    document = db.similarity_search(query, k=3);

    st.subheader("📒 Retrieved context");

    for i , doc in enumerate(document):
        st.markdown(f"**Result {i+1} : **");
        st.write(doc.page_content)