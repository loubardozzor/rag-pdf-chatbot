from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma, chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
import os

from openai.types.responses import response


load_dotenv()
open_ai_key = os.environ.get("OPENAI_API_KEY")

def text_processing(document:str)->list:
    """
        description: split a document into chunk
        args: 
            document: a text document we want to split into chunks
        return: a list of chunks
    """
    loader = PyPDFLoader(document)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    return chunks




def ingest_pdf_to_DB(chunks:list)->Chroma:
    """
        description: this function compute the embedding of every chunk and ingest it into chroma
        args:
            chunks: list of documents chunk
        return: the vector store
    """ 
    embedder =  OpenAIEmbeddings()
    return Chroma.from_documents(chunks,embedder )

    
    
def get_relevant_docs(query: str, vector_store: Chroma)->list:
    """
        description: this function find the most relevant document for the query
        args:
            query: query write by the user about its documents
            vector_store: the Chroma vector store instance

        return: a list of documents relevant to the user's query
    """
    return vector_store.similarity_search(query)




def get_response_llm(query: str, chunks:list)->str:
    """
        description: this function get the response to the user query  from the LLM passing to it the context
        args:
            query: user query write by the user about its documents
            chuncks: relevant documents to the user query getting from the vector db
        return :
                the answer formated as string from the user query      

    """

    context = " ".join([chunk.page_content for chunk in chunks]) # pythonic , more profesional
    template = ChatPromptTemplate.from_messages([
    ("system", "Sei un assistente utile che risponde usando solo il contesto fornito se la risposta alla domanda non si trova nel documento devi rispondere con la scritta 'non abbastanza elementi per rispondere a questa domanda."),
    ("user", "Contesto: {context}\nDomanda: {question}")])

    llm = ChatOpenAI(model="gpt-4",
                     temperature=0.2)
    chain = template|llm
    response = chain.invoke({"context":context, "question":query})
    return response


