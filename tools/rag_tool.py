from langchain.tools import tool
from tools.request_context import get_current_user_id

rag_instance = None


def set_rag_instance(rag):
    global rag_instance
    rag_instance = rag


@tool
def rag_search(query: str) -> str:
    """
    Use this tool to answer questions from uploaded documents or internal knowledge base.
    Use this when the question is related to user-provided files or stored data.
    Search is limited to the current end user's uploaded documents.
    """
    if rag_instance is None:
        return "RAG is not configured. Document search is unavailable."
    return rag_instance.query(query, user_id=get_current_user_id())