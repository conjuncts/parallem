import chromadb
from dotenv import load_dotenv
import parallem as pllm

# RAG implementation.
# parallem does not bundle any RAG libraries, but it can be implemented.

client = chromadb.Client()
collection = client.create_collection(name="rag_demo")

collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=[
        "Refunds are available within 30 days of purchase with a valid receipt.",
        "Digital products are non-refundable after download unless required by law.",
        "Our offices are based in Palo Alto, California.",
    ],
)


def vector_store_tool(query: str, k: int = 2) -> str:
    """Given a query, retrieves relevant documents from the vector store."""
    result = collection.query(query_texts=[query], n_results=k)
    docs = result["documents"][0]
    return "\n".join(docs)


# Begin parallem logic


def rag_agent(agt: pllm.AgentContext, query: str):
    conv = agt.get_msg_state()
    resp = conv.ask_llm(
        query,
        instructions="Only supply information relevant to the user's question.",
        tools=pllm.to_tool_schema([vector_store_tool]),
    )
    conv.ask_functions(vector_store_tool=vector_store_tool)
    conv.ask_llm()
    print(resp.function_calls)
    return conv[-1].final_answer


if __name__ == "__main__":
    load_dotenv()
    with pllm.resume_directory(
        ".pllm/example/rag", hash_by=["llm"], provider="google"
    ) as orch:
        with orch.agent() as agt:
            out = rag_agent(agt, "What is the refund policy for digital products?")
            print(out)
