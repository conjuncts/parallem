
By design, we do **not** offer a native API for retrieval augmented generation (RAG). 

There are many vector stores (Chroma, FAISS, Qdrant, Weaviate, Milvus, Pinecone, Elastisearch, pgvector), and we do not want to enforce a single vector store. 

Nor do we want to bundle a vector store library. To keep parallem **slim** and **lightweight**, RAG is outside our scope. However, RAG can still be easily accomplished with function calling. 

For example, interface with your vector store with a simple vanilla Python function.
Here is a minimal in-memory `chromadb` example:

```bash
pip install chromadb
```

```python
--8<-- "examples/advanced/simplest_rag.py"
```

In the above example, retrieval augmented generation is available as a function call. 
However, you can also feed the query directly to the vector store -- up to you.
```python
def simpler_rag_agent(agt: AgentContext, query: str)
    documents = vector_store_tool("What is the refund policy for digital products?")
    resp = agt.ask_llm(
        [*documents, query]
    )
    return resp.final_answer
```
