# Subagents

In the subagent pattern, a main agent creates child subagents to break down and complete a task in parallel. 

We support this pattern:

1. Create a parent agent.
2. Create a child agent, **no special syntax needed**.
3. (Optional) Give the child access to its parent's messages.

```python
--8<-- "examples/advanced/simplest_subagent.py"
```

Output:
```markdown
[INFO] Resuming with session_id=7
shape: (10, 2)
+-----------+-----------------------------------+
| place     | description                       |
| ---       | ---                               |
| str       | str                               |
+===============================================+
| Paris     | Paris is a city of light and l... |
| Tokyo     | Tokyo is a city where centurie... |
| New York  | New York City, often simply ca... |
| Rome      | Rome, the capital of Italy, is... |
| Barcelona | Barcelona sits on the northeas... |
| Istanbul  | Istanbul is a city where two c... |
| Bangkok   | Bangkok, Thailand's energetic ... |
| Sydney    | Sydney is Australia's largest ... |
| Kyoto     | Kyoto, in Japan's Kansai regio... |
| Bali      | Bali, Indonesia's famed island... |
+-----------+-----------------------------------+
```

## Aggregation

After creating subagents, you can aggregate their results.

Output:
```
Top 3 destinations for a mix of urban and countryside exploration:

- Kyoto, Japan — A city steeped in tradition with easy access to nature: Arashiyama’s bamboo grove, temple gardens, and the Philosopher’s Path let you wander from refined urban culture into serene countryside scenes, all in one trip.

- Sydney, Australia — A world-class city with immediate nature on your doorstep: iconic harbors and beaches (Bondi, Manly) plus coastal trails and nearby national parks give you city vibes and outdoor escapes in one headlining destination.

- Bali, Indonesia — Rich rural landscapes paired with vibrant culture: terraced rice paddies in Ubud, volcanic hikes (Mount Batur), and stunning temples sit alongside beach towns and lively markets, offering both countryside immersion and island-city flavor.
```

## Dynamically creating subagents

To create a subagent via a function call, pass the subagent declaration like any other function. Subagents will be created with the names passed into `subagent_names`.
