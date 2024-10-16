import streamlit as st
from huggingface_hub import login
from apikey import apikey
import os 
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores import Chroma
from PIL import Image
import chromadb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
login(token=apikey, add_to_git_credential=True)

def clear():
    if 'prompt' in st.session_state:
        st.session_state.prompt = ''

# Create two columns
col1, col2 = st.columns([1, 3])

# Load and display the image in the left column
with col1:
    image = Image.open('llama.png')
    st.image(image, width=100)  # Adjust width as needed

# Display the title in the right column
with col2:
    st.title('Chat-o-box Llama')
 
uploaded_file = st.file_uploader('Upload file: ', type=['pdf', 'docx', 'txt'])
col1,col2 = st.columns([1,3])
with col1:
    add_file = st.button('Upload File', on_click=clear)
    
with col2:
    st.button('Clear', on_click=clear)
if uploaded_file and add_file:
    bytes_data = uploaded_file.read()
    file_name = os.path.join('./', uploaded_file.name)
    with open(file_name, 'wb') as f:
        f.write(bytes_data)
    loader = TextLoader(file_name) # to load text document 
    documents = loader.load() 
    # print(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_documents(documents=documents)
    
    embeddings = OllamaEmbeddings(model="llama3.2")  
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    persist_directory = "./chroma_db"
    
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever()
    # docs = retriever.get_relevant_documents("Which family did Romeo belong to")
    # for i, doc in enumerate(docs, 1):
    #     with st.expander(f"Document {i}"):
    #         st.subheader("Content:")
    #         st.write(doc.page_content)


    llm = OllamaLLM(model="llama3.2")
    
    # Create the retrieval chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    if 'chain' not in st.session_state:
        st.session_state.chain = chain
    st.success('File uploaded, chunked and embedded successfully')
    

# template = """Question: {question}

# Answer: Let's think step by step."""

# template = "Please answer the following question: {question}\nProvide a detailed explanation in your answer."

# prompt = ChatPromptTemplate.from_template(template)
# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an expert in classic novels."),
#     ("human", "Please answer the following question: {question}"),
#     ("ai", "Here's the answer:")
# ])

# model = OllamaLLM(model="llama3.2")

# chain = prompt | model

prompt = st.text_input("Ask me something about the file:", key="prompt")
if prompt:
    with st.spinner("Generating..."):
        chain = st.session_state.chain
        question = {"question": prompt}
        answer = chain.invoke(question)
        st.write(answer)
