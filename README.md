
#  RAG-PDF for DnD documentation with FAISS + Ollama

This is a lightweight Retrieval-Augmented Generation (RAG) pipeline that:
- Loads PDF documents from the `data/` folder
- Splits them into text chunks
- Embeds them using an Ollama local embedding model
- Indexes them with FAISS
- Supports querying using a local LLM and embedded context

## Features

- Uses `nomic-embed-text` with Ollama for local embeddings
- Works offline
- Automatic loading of all PDFs in the `data/` folder
- Fast, in-memory retrieval using FAISS

##  Setup

1. Install Ollama and download models
   - Install from: https://ollama.com
   - Start Ollama server
   - Pull embedding model:
     ```bash
     ollama run nomic-embed-text
     ```

2. **Create environment and install dependencies**

   ```bash
   pip install -r requirements.txt
Add PDFs

Put one or more .pdf files inside the data/ folder.

## Run the indexer

bash

python db/populate_database.py --reset

## Query the documents
python db/query_data.py



## Example Query

Enter your question: What is the purpose of this document?