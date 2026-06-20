from rag import text_processing, ingest_pdf_to_DB, get_relevant_docs, get_response_llm
import streamlit as st

st.title("Chat with Your Documents")
uploaded_file = st.sidebar.file_uploader(label='carica il tuo PDF',
                        type =["pdf"])