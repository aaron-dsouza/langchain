from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFaceHub
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM
import streamlit as st
from huggingface_hub import login
from apikey import apikey
import os 

os.environ["TOKENIZERS_PARALLELISM"] = "false"
login(token=apikey, add_to_git_credential=True)
type="text-generation"

def clear():
    if 'prompt' in st.session_state:
        st.session_state.prompt = ''

st.button('Clear', on_click=clear)

# st.session_state.prompt = ''
# TODO Figure out the right model for question answering
if type == 'text-generation':
    model_id = "meta-llama/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe = pipeline(
        "text-generation", 
        model=model_id, 
        torch_dtype=torch.bfloat16, 
        # tokenizer=tokenizer,
        device_map="auto",
    )
else:
    model_id = "deepset/roberta-base-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, model_max_length=512)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id, torch_dtype=torch.float32)

    # Create the pipeline
    pipe = pipeline(
        "question-answering", 
        model=model, 
        torch_dtype=torch.float32, 
        tokenizer=tokenizer, 
        device_map="auto"
    )

# Store the pipeline in session state
if 'pipe' not in st.session_state:
    st.session_state.pipe = pipe

def generate_completion(prompt):
    pipe = st.session_state.pipe
    response = pipe(prompt, max_length=500, truncation=True)  # Adjust max_length as needed
    generated_text = response[0]['generated_text']
    return generated_text

def generate_answer(context, question):
    # Access the pipeline from session state
    pipe = st.session_state.pipe
    # Generate answer based on context and question
    response = pipe(question=question, context=context)
    print(response)
    return response['answer']


st.title('Chatterbox') 
if type == 'text-generation':
    prompt = st.text_input("Say something:", key="prompt")
    if prompt:
        with st.spinner("Generating..."):
            completion = generate_completion(prompt)
            st.success("Completion:")
            st.write(completion)
else:
    # Predefined context
    context = (
        "You are an expert in classical novels, including works like 'Pride and Prejudice', 'Moby Dick', 'A Tale of Two Cities', and many others. "
        "You have extensive knowledge of their plots, themes, and characters."
    )
    question = st.text_input("Ask me something:", key="prompt")

    if question:
        with st.spinner("Generating..."):
            answer = generate_answer(context, question)
            st.success("Answer:")
            st.write(answer)