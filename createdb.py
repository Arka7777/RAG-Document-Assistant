#load pdf
#split into chunks
#create emb
#store into chroma

from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

loader=PyPDFLoader("doc loaders/GRU.pdf")
data=loader.load()

splitter=RecursiveCharacterTextSplitter(
  chunk_size=1000,
  chunk_overlap=200 
)

chunks=splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorStore=Chroma.from_documents(
  documents=chunks,
  embedding=embeddings,
  persist_directory="chroma-db"
)
