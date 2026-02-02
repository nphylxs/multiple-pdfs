import streamlit as st

def main():
    st.set_page_config(page_title="Multiple PDFs Uploader", page_icon="📖")
    st.header("📖 Multiple PDFs Uploader")
    st.chat_input("Ask questions about your PDFs!")

    with st.sidebar:
        st.subheader("Upload your documents")
        uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
        st.button("Upload")
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files successfully!")
    
if __name__ == "__main__":
    main()