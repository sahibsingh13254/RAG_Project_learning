from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -----------------------------
# Create sample documents
# -----------------------------
docs = [
    Document(page_content="Gradient descent is an optimization algorithm used to minimize loss."),
    Document(page_content="Gradient descent minimizes the loss function by updating weights."),
    Document(page_content="Gradient descent is an optimization technique used in machine learning."),
    Document(page_content="Neural networks are trained using gradient descent and backpropagation."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# -----------------------------
# Create Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Create Vector Store
# -----------------------------
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings
)

# -----------------------------
# Similarity Retriever
# -----------------------------
similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# -----------------------------
# MMR Retriever
# -----------------------------
mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "fetch_k": 5,
        "lambda_mult": 0.5
    }
)

query = "What is gradient descent?"

print("===== Similarity Search =====")
similarity_docs = similarity_retriever.invoke(query)

for i, doc in enumerate(similarity_docs, start=1):
    print(f"\nDocument {i}")
    print(doc.page_content)

print("\n=============================\n")

print("===== MMR Search =====")
mmr_docs = mmr_retriever.invoke(query)

for i, doc in enumerate(mmr_docs, start=1):
    print(f"\nDocument {i}")
    print(doc.page_content)