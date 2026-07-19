from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

data = PyPDFLoader ("document_loaders/AI_Learning_Roadmap.md.pdf")

docs = data.load()

splitter = RecursiveCharacterTextSplitter(
     chunk_size=40,
    chunk_overlap=5,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_documents(docs)

 # print(len(chunks))
print(chunks[0].page_content)