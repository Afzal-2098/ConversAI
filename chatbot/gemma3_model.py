from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

# Load environment variables from .env file
load_dotenv()


# Set environment variables for LangSmith
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")

llm = OllamaLLM(model="gemma3:1b")

# Streamlit app
st.title("Gemma3 Chatbot")
st.write("This is a simple chatbot using the Gemma3 model.")
st.write("Ask me anything!")

input_value = st.text_input("Enter your question:", key="input", placeholder="Type your question here...")
if input_value:
    response = llm.invoke(input_value)
    st.write("Response:", response)