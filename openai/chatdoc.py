import os
from apikey import apikey
import streamlit as st
from langchain_openai import ChatOpenAI # used for GPT3.5/4 model from langchain_community.document_loaders import TextLoader from langchain.text_splitter import RecursiveCharacterTextSplitter from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain


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
    print(documents) # print to ensure document loaded correctly.

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_documents(documents=documents)

    # st.write(chunks[0])
    # st.write(chunks[1])

    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    retriever=vector_store.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
    st.session_state.chain = chain
    st.success('File uploaded, chunked and embedded successfully')

question=st.text_input('Input your question', key='question')

if question:
    if 'chain' in st.session_state:
        chain = st.session_state.chain
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        response = chain.invoke({'question':question, 'chat_history':st.session_state['history']})
        st.session_state['history'].append((question, response['answer']))
        st.write(response['answer'])
        st.header('Chat history')
        for prompts in st.session_state['history']:
            st.write("Question: "+prompts[0])
            st.write("Answer: "+prompts[1])
    