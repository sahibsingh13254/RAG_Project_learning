from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader ("document_loaders/AI_Learning_Roadmap.md.pdf")

docs = data.load()

print(len(docs))