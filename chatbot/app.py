from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os
from langchain_community.llms import ollama

# Load environment variables from .env file
load_dotenv()

# Set environment variables for DeepSeek
os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")

# Set environment variables for LangSmith
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")

llm = ChatDeepSeek(model="deepseek-chat")
response = llm.invoke("Hello, world!")
print(response)