# Subagents

The **subagent pattern** is when a main agent creates many child subagents in order to break down and complete a task in parallel. 

This pattern is supported by parallem. Simply create an agent from within an existing agent context block.

Core pattern:

1. Create a parent conversation.
2. For each item, create a new named agent.
3. Copy parent context into each child with `sub_conv.extend(parent_conv)`.
4. Ask a focused question in each child and collect results.

```python
--8<-- "examples/advanced/simplest_subagent.py"
```

```markdown
[INFO] Resuming with session_id=5
['Paris', 'Tokyo', 'New York', 'Rome', 'Barcelona', 'Istanbul', 'Bangkok', 'Sydney', 'Kyoto', 'Bali']
1. Paris is a city of light and layers, whe...
2. Tokyo is a city where centuries of tradi...
3. New York City, often simply called New Y...
4. Rome, the capital of Italy, is a living ...
5. Barcelona sits on the northeastern coast...
6. Istanbul is a city where two continents ...
7. Bangkok, Thailand's energetic capital, s...
8. Sydney is Australia's largest city, fame...
9. Kyoto, in Japan's Kansai region, long se...
10. Bali, Indonesia's famed island, lies bet...
```

## Dynamically creating subagents

You may want another type of subagent, where the parent agent creates a subagent via a function call.

For this pattern, simply pass a function (which takes a `pllm.AgentContext` an argument) as a regular function call. ParaLLeM will automatically inject it with an agent. But you will also need to pass a name for these newly created agents to `subagent_names`.

```python
import parallem as pllm
from dotenv import load_dotenv


def city_summary_agent(agt: pllm.AgentContext, city: str) -> str:
	"""Creates an up-to-date summary of a city as a travel destination."""
	conv = agt.get_msg_state()
	conv.ask_llm(
		f"Write one concise paragraph about {city} as a travel destination."
	)
	return conv[-1].final_answer


load_dotenv()
with pllm.resume_directory(".pllm/example/subagent-dynamic", dashboard=True) as orch:
	with orch.agent("planner") as agt:
		conv = agt.get_msg_state()
		conv.ask_llm(
			"Generate summaries of 3 popular travel destinations.",
			tools=pllm.to_tool_schema([city_summary_agent]),
		)

		fc_outs = conv.ask_functions(
			city_summary_agent=city_summary_agent,
			subagent_names=["city-agent-1", "city-agent-2", "city-agent-3"],
		)

		conv.ask_llm(
			"Combine these city summaries into a ranked list.",
			*fc_outs,
		)
		print(conv[-1].final_answer)
```

If fewer names are provided than required injected calls, `ask_functions` raises a `ValueError`.

## See also

- [Aggregation](aggregation.md)