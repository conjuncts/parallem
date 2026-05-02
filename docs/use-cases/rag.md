
By design, we do **not** offer a native API for retrieval augmented generation (RAG). 

Of the many vector stores (Chroma, FAISS, Qdrant, Weaviate, Milvus, Pinecone, Elastisearch, pgvector), we do not want to enforce just one, and we don't want to bundle a vector store library. To keep parallem lightweight, we think RAG is outside our scope. 

Nonetheless, you can access this functionality via function calling. For example, here's an in-memory `chromadb` example:

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
