ParaLLeM excels for processing documents in high throughput. 

By switching to batch mode, you save 50% on token costs, save CPU time, and can scale up to 1000s of documents.

In this example, we count the number of syllables in a [word list](https://github.com/zydou/high-frequency-words/raw/refs/heads/master/100k.txt) of 100k words. 

!!! warning

    The following example makes 100k (!) LLM requests. It costs ~$0.50 to complete with OpenAI. It uses "max_output_tokens=20" to save costs, but this is specific to OpenAI! Make sure to adjust `kwargs` for other providers. Not constraining reasoning can lead to costs of >$100. Use other providers at your own risk. 

```python title="examples/stress/stress_test.py"
--8<-- "examples/stress/stress_test.py"
```
