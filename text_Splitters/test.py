from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator  = "",
    chunk_size =10,
    chunk_overlap =1,
)

data = TextLoader("text_splitters/characters.txt")

docs = data.load()

chunks = splitter.split_documents(docs)

print(docs)

for i in chunks:
    print(i.page_content)
    print()
    print(repr(i.page_content), len(i.page_content))