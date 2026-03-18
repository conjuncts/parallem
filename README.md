# pipelinellm

## Compatibility

| Sync/Batch | OpenAI | Anthropic | Google |
| --- | --- | --- | --- |
| Simple | ✅ | ✅ | ✅ |
| Structured Output | ✅ | ✅ | ✅ |
| Function Calls | ✅ | ✅ | ✅ |
| Web Search | ✅ | ✅ | ✅ |
| Image Input | ✅ | ✅ | ✅ |
| MCP | ✅ | - | - |


## Philosophy

1. A library designed with the Batch API in mind.
2. We aim to support pipelines where LLMs are "input/output machines", rather than an interactive conversational agent.
3. LLM pipeline control flow should be represented with Python, rather than data structures.
4. Circumvent vendor lock-in and "architecture lock-in". (Model agnostic and architecture agnostic.)
5. Effortless parallelization.
6. Improved Developer Experience (develop and debug as synchronous, quickly scale up to huge pipelines).