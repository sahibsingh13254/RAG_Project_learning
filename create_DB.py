# load pdf
# split into chunks
# create the embeddings
# Store into chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("document_loaders/AI_Learning_Roadmap.md.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=5,
)

chunks = splitter.split_documents(docs)

embedding_model = MistralAIEmbeddings()
vectorstore= Chroma.from_documents(
    documents = chunks,
    persist_directory = "chroma_database"

)