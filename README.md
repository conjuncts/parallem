# pipelinellm

(Parallel Language Models) *p*-LLM



## Compatibility


| Default | Sync | Concurrent* | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ |
| Google | ✅ | ✅ | ✅ |


| Structured Output | Sync | Concurrent | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ❌ | ❌ | ❌ |
| Google | ✅ | ✅ | ✅ |

| Function Calls | Sync | Concurrent | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ |
| Google | ✅ | ✅ | ✅ |

| Web Search | Sync | Concurrent | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ |
| Google | ✅ | ✅ | ✅ |

| Image Input | Sync | Concurrent | Batch |
| --- | --- | --- | --- |
| OpenAI | ✅ | ✅? | ✅ |
| Anthropic | ✅ | ✅? | ✅ |
| Google | ✅ | ✅? | ✅ |

*Concurrent is similar to async, but it isn't true async because `await` and `.resolve()` doesn't actually yield control to other processes. It does allow parallelization and requests to be run concurrently.

## Philosophy

1. A library designed with the Batch API in mind.
2. We aim to support pipelines where LLMs are "input/output machines", rather than an interactive conversational agent.
3. LLM pipeline control flow should be represented with Python, rather than data structures (ie. LangChain, LangGraph).
4. Circumvent Vendor Lock-in and "Architecture Lock-in".
5. Effortless parallelization.
6. Improved Developer Experience (Develop and debug as synchronous, quickly scale up to huge pipelines).