from langchain_community.document_loaders import TextLoader

loader=TextLoader("notes.txt")
data=loader.load()
print(data[0].page_content)
print(len(data))