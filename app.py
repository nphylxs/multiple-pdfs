import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS


def get_pdf_texts(pdfs):
    texts = "" 
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            texts += page.extract_text() + "\n"
    return texts

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        )
    return text_splitter.split_text(text)

def vector_store_from_chunks(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vector_store

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDFs Uploader", page_icon="📖")
    st.header("📖 Multiple PDFs Uploader")
    st.chat_input("Ask questions about your PDFs!")

    with st.sidebar:
        st.subheader("Upload your documents")
        pdfs = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        if pdfs:
            st.success(f"Uploaded {len(pdfs)} files successfully!")
        if st.button("Process"):
            with st.spinner("Processing PDFs..."):
                raw_texts = get_pdf_texts(pdfs)
                
                text_chunks = get_text_chunks(raw_texts)

                vector_store = vector_store_from_chunks(text_chunks)

                
    
if __name__ == "__main__":
    main()