import streamlit as st
import tempfile
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.vectorstores import Chroma

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="PDF RAG Analyzer (Ollama + Chroma)", layout="wide")
st.title("🦙 PDF Analyzer using RAG (LLaMA-3 + Chroma)")
st.markdown(
    """
Upload a PDF and ask questions.  
✅ Fully offline with LLaMA-3 via Ollama  
✅ Works on Python 3.13  
✅ No C++ compiler required
"""
)

# ------------------------------
# PDF UPLOAD
# ------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing PDF..."):

        # Save uploaded PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(documents)

        # Embeddings (Ollama)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

        # Vector store (Chroma)
        vectorstore = Chroma.from_documents(chunks, embeddings, collection_name="pdf_rag")

        # LLM (LLaMA-3 via Ollama)
        llm = Ollama(model="llama3", temperature=0)

        # RAG Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

        st.success("✅ PDF processed successfully!")

    # ------------------------------
    # QUESTION INPUT
    # ------------------------------
    query = st.text_input("Ask a question from the PDF")

    if query:
        with st.spinner("Analyzing PDF..."):
            result = qa_chain(query)

            st.subheader("📌 Answer")
            st.write(result["result"])

            st.subheader("📄 Source Chunks")
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:500] + "...")

    # Cleanup
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
