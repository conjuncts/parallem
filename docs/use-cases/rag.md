
By design, pipelinellm does **not** offer a native API for retrieval augmented generation (RAG). This is because there are many vector stores (Chroma, FAISS, Qdrant, Weaviate, Milvus, Pinecone, Elastisearch, pgvector). 

We do not want to enforce a single way, and neither do we want to bundle a vector store library. Pipelinellm should be slim and lightweight, and native RAG is outside the scope of our library. However, it can still be easily accomplished with functions and/or function calls. 

For example, interface with your vector store with a **simple vanilla Python function**.
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
