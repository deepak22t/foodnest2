from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
def get_llm():
    return ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )