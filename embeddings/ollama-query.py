from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

persist_directory = "./chroma_db"  # The same directory you used to save
embeddings = OllamaEmbeddings(model="llama3.2")

# Load the persisted database
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Now you can use the loaded database
retriever = db.as_retriever()

docs = retriever.get_relevant_documents("Which family did Romeo belong to")
print(len(docs))
for doc in docs:
    print(doc.page_content)