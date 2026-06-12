from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
open_ai_key = os.environ.get("OPEN_API_KEY")
