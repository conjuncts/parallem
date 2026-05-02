
By design, we do **not** offer a native API for managing unique user conversations. 

Of the many ways to do this (Redis, PostgreSQL, MongoDB), we don't want to enforce just one method, and we don't want to bundle a database. To keep ParaLLeM lightweight, we think that storing conversation IDs is outside our scope. 

Nonetheless, you can access this functionality by assigning each conversation a unique `conversation_uuid` as the agent name. The following demonstrates a simple CLI chatbot, storing `conversation_uuid`'s to disk with polars:

```python title="examples/advanced/local_chatbot.py"
--8<-- "examples/advanced/local_chatbot.py"
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