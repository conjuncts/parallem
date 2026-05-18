ParaLLeM scales up to millions of requests. This makes it excellent for large data pipelines.

!!! warning

    The following example makes 1M (!) LLM requests. It costs ~$5 to complete with OpenAI. It uses "max_output_tokens=20" to save costs, but this is specific to OpenAI! Make sure to adjust `kwargs` for other providers. Not constraining reasoning can lead to costs of >$100. Use other providers at your own risk. 

```python
--8<-- "examples/stress/stress_1m.py"
```