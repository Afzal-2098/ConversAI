from dotenv import load_dotenv
import os
# from langchain_ollama import OllamaLLM
import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

# Load environment variables from .env file
load_dotenv()


# Set environment variables for LangSmith
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["QWEN3_API_KEY"] = os.getenv("QWEN3_API_KEY")

# llm = init_chat_model(model="gemma3:1b", model_provider="ollama")
llm = init_chat_model(model="qwen3-0.6b-04-28:free", model_provider="Groq")

# Define a New grapg
workflow = StateGraph(
    state_schema=MessagesState
)


# Define the function that calls the model
def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

# Streamlit app
st.title("Gemma3 Chatbot")
st.write("This is a simple chatbot using the Gemma3 model.")
st.write("Ask me anything!")

input_value = st.text_input("Enter your question:", key="input", placeholder="Type your question here...")
input_messages = HumanMessage(content=input_value)
if input_value:
    response = app.invoke({"messages": [input_messages]}, config=config)
    output_messages = response["messages"][-1]
    st.write("Response: \n\n", output_messages.content)