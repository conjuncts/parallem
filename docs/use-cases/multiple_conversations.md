
By design, pipelinellm does **not** offer a native API for managing unique conversation IDs. There are many ways to do this (Redis, Postgres), and we don't want to enforce just one way.

We wish to keep pipelinellm slim and lightweight, and we believe that storing conversation IDs is outside the scope of our library. Instead, it is a task for your backend to handle. 

Nonetheless, it can be easily accomplished. After assigning a unique conversation uuid, assign the uuid to be the agent name. The following demonstrates a simple CLI chatbot using polars:

```python
--8<-- "examples/advanced/simplest_multiconv_chatbot.py"
```


Output:

<div class="highlight session-log-html"><pre><code>[<span class="log-tag">INFO</span>] Resuming with session_id=24

Existing conversations
======================
<span class="log-tag">0</span>: "Who was president before Lincoln?"
<span class="log-tag">1</span>: "What is the capital of Kyrgyzstan?"
<span class="log-tag">2</span>: [New conversation]
<span class="log-tag">Enter</span> to quit
Select a number:
</code></pre></div>