import os
import time
from agent.agent_builder import sales_agent
from tools.rag import RAGPipeline
from tools.rag_tool import set_rag_instance
from tools.request_context import current_user_id

def main():
    rag = RAGPipeline()
    set_rag_instance(rag)
    agent = sales_agent()

    uid = os.getenv("CLI_USER_ID", "default")
    print(f"🔥 AI Agent Ready (type 'exit' to quit) — RAG user_id={uid!r} (set CLI_USER_ID in env to change)\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        start_time = time.time()
        token = current_user_id.set(uid)
        try:
            response = agent.invoke({
                "messages": [
                    {"role": "user", "content": query}
                ]
            })
        finally:
            current_user_id.reset(token)

        elapsed_time = time.time() - start_time

        print(f"\nAI: {response['messages'][-1].content}")
        print(f"⏱️ Processed in {elapsed_time:.2f} seconds\n")


if __name__ == "__main__":
    main()