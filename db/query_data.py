# query_data.py
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import sys

# Load user query
query = sys.argv[1] if len(sys.argv) > 1 else input("Enter your question: ")

# Load FAISS index
db = FAISS.load_local(
    "faiss_index",
    embeddings=OllamaEmbeddings(model="nomic-embed-text"),
    allow_dangerous_deserialization=True
)

# Initialize Retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Load LLM (adjust as per model)
llm = Ollama(model="llama3.1")

# Build QA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Get answer
result = qa.run(query)
print(f"\n Answer: {result}")