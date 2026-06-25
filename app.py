from rag import text_processing, ingest_pdf_to_DB, get_relevant_docs, get_response_llm
import streamlit as st

@st.cache_resource
def embedding_chunks(document_name:str):
    """ description : this is a wrapper function that help us to embedd a doc pdf one the program is launched
        args: 
            document_name: the filename of the document to embbedù
        return:
              return a vector db that store the embedding of the document
    """
    chunks = text_processing(document_name)
    chroma_db = ingest_pdf_to_DB(chunks)
    return chroma_db


st.title("Chat with Your Documents")
uploaded_file = st.sidebar.file_uploader(label='carica il tuo PDF',
                        type =["pdf"])

if uploaded_file is not None:
    # procssare il file 
    with open(uploaded_file.name, "wb") as f: # apriamo un file in modalta scrittura col nome del file caricato su streamlit e copiamo poi il sui contenuto nel file 
        f.write(uploaded_file.getbuffer())
        chroma_db = embedding_chunks(uploaded_file.name)
query = st.text_input("parla coi tuoi documenti facendo una domanda")
if query: # la query non è una stringa vuota
    more_relevant_docs = get_relevant_docs(query, chroma_db)
    response = get_response_llm(query, more_relevant_docs)
    st.write(response.content)

