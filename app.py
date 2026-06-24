from rag import text_processing, ingest_pdf_to_DB, get_relevant_docs, get_response_llm
import streamlit as st

st.title("Chat with Your Documents")
uploaded_file = st.sidebar.file_uploader(label='carica il tuo PDF',
                        type =["pdf"])

if uploaded_file is not None:
    # procssare il file 
    with open(uploaded_file.name, "wb") as f: # apriamo un file in modalta scrittura col nome del file caricato su streamlit e copiamo poi il sui contenuto nel file 
        f.write(uploaded_file.getbuffer())
    chunks = text_processing(uploaded_file.name)
    chroma_db = ingest_pdf_to_DB(chunks)
query = st.text_input("parla coi tuoi documenti facendo una domanda")
if query: # la query non è una stringa vuota
    more_relevant_docs = get_relevant_docs(query, chroma_db)
    response = get_response_llm(query, more_relevant_docs)

st.write(response.content)