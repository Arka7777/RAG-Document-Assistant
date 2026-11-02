from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Python is widely used for artificial intelligence and machine learning.",
        metadata={"topic": "Python"}
    ),
    Document(
        page_content="React is a JavaScript library used to build interactive user interfaces.",
        metadata={"topic": "React"}
    ),
    Document(
        page_content="RAG allows language models to retrieve information from external documents before generating an answer.",
        metadata={"topic": "RAG"}
    )
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

vectorStore=Chroma.from_documents(
  documents=docs,
  embedding=embeddings,
  persist_directory="chroma-db"
)

res=vectorStore.similarity_search("what is used for interactive user interfaces",k=2)

for r in res:
  print(r)
