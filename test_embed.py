from langchain_community.embeddings import OllamaEmbeddings

embedder = OllamaEmbeddings(model="nomic-embed-text")
res = embedder.embed_documents(["Hello world"])
print(res)