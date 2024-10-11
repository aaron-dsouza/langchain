import os
from apikey import apikey
import streamlit as st
# import numpy as np
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import chromadb
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer

from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline
# from langchain.chains import ConversationalRetrievalChain
# /Users/ironman/Code/langchain/huggingface/chatdoc.py:40: LangChainDeprecationWarning: 
# The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. 
# An updated version of the class exists in the :class:`~langchain-huggingface package and should be used instead. 
# To use it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:`~langchain_huggingface import HuggingFaceEmbeddings``.
#   embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Choose the model you want to use


os.environ["OPENAI_API_KEY"] = apikey

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
        st.session_state.question = ''

st.title('Chat with Document') 
add_file = st.button('Clear History', on_click=clear_history)
uploaded_file = st.file_uploader('Upload file: ', type=['pdf', 'docx', 'txt'])
add_file = st.button('Add File', on_click=clear_history)
if uploaded_file and add_file:
    bytes_data = uploaded_file.read()
    file_name = os.path.join('./', uploaded_file.name)
    with open(file_name, 'wb') as f:
        f.write(bytes_data)
    loader = TextLoader(file_name) # to load text document 
    documents = loader.load() 
    # print(documents) # print to ensure document loaded correctly.

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_documents(documents=documents)
    # Define the path to the pre-trained model you want to use
    modelPath = "sentence-transformers/all-MiniLM-l6-v2"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    model_kwargs = {'device':'cpu'}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings_model = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )

    text = "This is a test document."
    query_result = embeddings_model.embed_query(text)
    query_result[:3]
    db = FAISS.from_documents(chunks, embeddings_model)
    # embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Choose the model you want to use
    # st.session_state.embeddings_model = embeddings_model
    # # client = chromadb.Client()
    # vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings_model)
    # st.session_state.vector_store = vector_store
    # llm = HuggingFacePipeline.from_model_id(model_id="gpt2", task="text-generation")
    # chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vector_store.as_retriever()
    # )

    # st.session_state.chain = chain
    st.success('File uploaded, chunked and embedded successfully')
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
question=st.text_input('Input your question', key='question')

if question:
    if 'chain' in st.session_state:
        chain = st.session_state.chain
        vector_store = st.session_state.vector_store
        context = vector_store.similarity_search(question, k=1)
        embeddings_model=st.session_state.embeddings_model
        max_context_tokens = 512
        context_text = context[0].page_content 
        inputs = tokenizer(question, context_text, return_tensors='pt', truncation=True, max_length=512, padding=True)

        if 'history' not in st.session_state:
            st.session_state['history'] = []
        # answer = chain.run(question=question, max=2000)
        # st.write(answer)
        input_data = {
            "query": question,
            "chat_history": st.session_state['history'],
            "context": inputs["attention_mask"]
        }
        response = chain.invoke(input=input_data, max_new_tokens=200)
        st.session_state['history'].append((question, response['answer']))
        st.write(response['answer'])
        # st.header('Chat history')
        # for prompts in st.session_state['history']:
        #     st.write("Question: "+prompts[0])
        #     st.write("Answer: "+prompts[1])
    