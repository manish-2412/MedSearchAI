from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Paths
DATA_PATH = '/Users/manish/Downloads/DiagnoAI-main/data/'  # Make sure this path points to the correct folder
DB_FAISS_PATH = '/Users/manish/Downloads/DiagnoAI-main/vectorstore/db_faiss'  # Path to store the FAISS database

# Create vector database
def create_vector_db():
    if not os.path.exists(DATA_PATH):
        print(f"Data folder {DATA_PATH} does not exist.")
        return

    # Load the documents from the specified folder
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        print("No documents found.")
        return

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Use HuggingFace embeddings for creating vector representations of the documents
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})

    # Create and save the FAISS vector database
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector database created and saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
