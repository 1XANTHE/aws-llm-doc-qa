from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load the dataset
loader = TextLoader("data/aws_docs.txt")
documents = loader.load()

# Create embeddings model
embeddings = HuggingFaceEmbeddings()

# Convert documents to vector database
vectorstore = FAISS.from_documents(documents, embeddings)

# Save the vector database
vectorstore.save_local("vector_store")

print("Vector database created successfully!")
