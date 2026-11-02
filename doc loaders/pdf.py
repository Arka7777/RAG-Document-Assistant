from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader("GRU.pdf")

docs=loader.load()
