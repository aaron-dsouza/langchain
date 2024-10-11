from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFaceHub
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import json
import streamlit as st

# TODO Figure out the right model for question answering
model_id = "meta-llama/Llama-3.2-1B"
model_id = "deepset/roberta-base-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForQuestionAnswering.from_pretrained(model_id, torch_dtype=torch.float32)

# Create the pipeline
pipe = pipeline("question-answering", model=model, torch_dtype=torch.float32, 
                tokenizer=tokenizer, device_map="auto")

# pipe = pipeline(
#     "question-answering", 
#     model=model_id, 
#     torch_dtype=torch.float32, 
#     device_map="auto"
# )
# Store the pipeline in session state
if 'pipe' not in st.session_state:
    st.session_state.pipe = pipe

def generate_completion(prompt):
    pipe = st.session_state.pipe
    response = pipe(prompt, max_length=500)  # Adjust max_length as needed
    generated_text = response[0]['generated_text']
    return generated_text

def generate_answer(context, question):
    # Access the pipeline from session state
    pipe = st.session_state.pipe
    # Generate answer based on context and question
    response = pipe(question=question, context=context)
    print(response)
    return response['answer']

# Predefined context
context = (
    "You are an expert in classical novels, including works like 'Pride and Prejudice', 'Moby Dick', 'A Tale of Two Cities', and many others. "
    "You have extensive knowledge of their plots, themes, and characters."
)
st.title('Chatterbox') 
question = st.text_input("Enter your prompt here:", key="prompt")

if question:
    with st.spinner("Generating..."):
        answer = generate_answer(context, question)
        st.success("Answer:")
        st.write(answer)