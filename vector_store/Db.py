from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

from langchain_core.documents import Document

docs = [
    Document(
        page_content="Java is an object-oriented programming language that follows the Write Once Run Anywhere principle.",
        metadata={"source": "java_book"}
    ),

    Document(
        page_content="Spring Boot simplifies Java application development by providing auto-configuration and embedded servers.",
        metadata={"source": "spring boot"}
    ),
     Document(
        page_content="REST APIs use HTTP methods such as GET, POST, PUT, and DELETE to communicate between client and server.",
        metadata={"source": "rest api"}
    ),

    Document(
        page_content="Hibernate is an ORM framework that maps Java objects to database tables.",
        metadata={"source": "hibernate"}
    )
]

embedding_model = MistralAIEmbeddings()

vectorstore = Chroma.from_documents(
    documents = docs,
    embedding = embedding_model,
    persist_directory = "chroma-db" 

)

result = vectorstore.similarity_search("what is springboot used for?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)

retriever = vectorstore.as_retriever()
docs = retriever.invoke("what is java")

for d in docs:
     print(r.page_content)

