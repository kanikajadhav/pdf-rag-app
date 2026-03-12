import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

st.title("📄 Chat with your PDF")
st.write("Upload a PDF and ask questions about it!")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        temp_path = f.name

    loader = PyPDFLoader(temp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the context below:
    
    Context: {context}
    
    Question: {question}
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    st.success("✅ PDF loaded! Ask me anything about it.")

    question = st.text_input("Ask a question:")
    if question:
        with st.spinner("Thinking..."):
            answer = chain.invoke(question)
            st.write("**Answer:**", answer)

    os.unlink(temp_path)