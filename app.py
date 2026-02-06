import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

from styles import css, bot_template, user_template


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

def get_convo_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",temperature=1.0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_user_input(user_input):
    response = st.session_state.conversation({
        'question': user_input,
    })
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Multiple PDFs Uploader", page_icon="📖")
    st.write(css, unsafe_allow_html=True)
    st.header("📖 Multiple PDFs Uploader")
    user_input = st.chat_input("Ask questions about your PDFs!")

    if user_input:
        handle_user_input(user_input)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.write(user_template.replace("{{MSG}}","Hi"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)
    
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

                st.session_state.conversation = get_convo_chain(vector_store)
                
if __name__ == "__main__":
    main()