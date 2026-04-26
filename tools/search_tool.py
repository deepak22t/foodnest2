from langchain_tavily import TavilySearch

def get_search_tool():
    return TavilySearch(
        max_results=3,
        search_depth="basic"   # fast response
    )