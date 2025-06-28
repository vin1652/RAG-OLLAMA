import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "nomic-embed-text"
DATA_DIR = "data"

def main():
    print("Loading documents from 'data/' folder...")
    pdf_paths = list(Path(DATA_DIR).glob("*.pdf"))

    if not pdf_paths:
        print("No PDF files found in 'data/' folder.")
        return

    docs = []
    for path in pdf_paths:
        print(f"📄 Loading: {path.name}")
        loader = PyPDFLoader(str(path))
        docs.extend(loader.load())

    print(f"📄 Loaded {len(docs)} pages total.")

    print(" Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Total chunks: {len(chunks)}")

    print("Embedding using Ollama...")
    embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

    print("Indexing with FAISS...")
    db = FAISS.from_documents(chunks, embedding=embedding)
    db.save_local("faiss_index")
    print("Success: FAISS in-memory index created!")

if __name__ == "__main__":
    main()
