from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()
open_ai_key = os.environ.get("OPENAI_API_KEY")

def text_processing(document):
    """
        description: split a document into chunk
        args: document a test document we want to split into chunks
        return: a list of chunks
    """
    loader = PyPDFLoader(document)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    return chunks



