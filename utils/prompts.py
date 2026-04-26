from langchain_core.prompts import ChatPromptTemplate

def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", """You are an intelligent AI assistant.

Rules:
- Use tools when required
- Use web search for latest or unknown info
- Do NOT hallucinate
- Keep answers short and accurate
"""),
        ("user", "{input}")
    ])