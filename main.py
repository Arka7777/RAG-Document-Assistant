from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings



template=ChatPromptTemplate.from_messages(
  [("system","""You are a helpful ai assistant.
  Use only the provided context to answrer the question.
  If the answer is not present in the content ,
  say: I could not find the answer in the document."""),
   ("human","""
Context:{context}
Question:{question}""")]
)

emb_model=GoogleGenerativeAIEmbeddings(
  model="gemini-embedding-001"
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

vectorstore=Chroma(
  persist_directory="chroma-db",
  embedding_function=emb_model
)

retriever=vectorstore.as_retriever(
  search_type="mmr",
  search_kwargs={
    "k":4,
    "fetch_k":10,
    "lambda_mult":0.5

  }
)

print("Rag system created")
print("press 0 to exit")

while True:
  query=input("you : ")
  if(query=="0"):
    break

  docs=retriever.invoke(query)

  context= "\n\n".join(
    [doc.page_content for doc in docs]
  )

  final_prompt=template.invoke({
    "context":context,
    "question":query
  })

  res=model.invoke(final_prompt)
  print(f"\n AI : {res.content}")



