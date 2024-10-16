from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
import time

loader = TextLoader("./romeo-and-juliet.txt") # to load text document 
documents = loader.load() 
print("document loaded")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks = text_splitter.split_documents(documents=documents)
print("Chunks created")
embeddings = OllamaEmbeddings(model="llama3.2")  
chromadb.api.client.SharedSystemClient.clear_system_cache()
start_time = time.time()
print("Creating embeddings")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
end_time = time.time()

execution_time = end_time - start_time

print("Embeddings created successfully!")
print(f"Time to create embeddings: {execution_time} seconds")