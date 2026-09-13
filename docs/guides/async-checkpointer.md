# Asynchronous Checkpointing

The [`AsyncRiftCheckpointSaver`](../reference/checkpointer.md) provides high-throughput, non-blocking multiversal state management for async agent pipelines and web servers (e.g. FastAPI, Starlette, Quart).

---

## ⚡ Basic Async Setup

```python
import asyncio
from langgraph.graph import StateGraph
from riftpoint import AsyncRiftCheckpointSaver

saver = AsyncRiftCheckpointSaver()
graph = builder.compile(checkpointer=saver)


async def main():
    config = {"configurable": {"thread_id": "async-session-1"}}
    result = await graph.ainvoke({"query": "analyze dataset"}, config=config)
    print("Async result:", result)


asyncio.run(main())
```

---

## 🚀 High-Throughput Speculative Pipelines

`AsyncRiftCheckpointSaver` pairs with [`RiftRunner.arun_parallel_branches`](../reference/runner.md) to dispatch dozens of candidate agent futures concurrently using `asyncio.gather`:

```python
from riftpoint import RiftRunner, MultiverseResolver, BranchSpec, HeuristicEvaluator

runner = RiftRunner(saver=saver)
resolver = MultiverseResolver(saver=saver)

# Concurrently run 20 async candidate tool branches
specs = [
    BranchSpec(name=f"candidate_{i}", input_data={"strategy_id": i}) for i in range(20)
]

# Non-blocking parallel execution
results = await runner.arun_parallel_branches(
    graph=graph,
    initial_config={"configurable": {"thread_id": "root-thread"}},
    branch_specs=specs,
)

# Non-blocking collapse and prune
collapse_result = await resolver.acollapse(
    thread_id="root-thread",
    results=results,
    evaluator=HeuristicEvaluator(scorer=lambda r: float(r.output["score"])),
    prune_discarded=True,
)
```

---

## 📈 Performance & Throughput

As measured in our [Benchmarks Report](../BENCHMARKS.md):

- **`aput()` Throughput:** $\approx 67,000\text{ ops/s}$ with mean latency of $12.4\,\mu\text{s}$.
- **`aget_tuple()` Throughput:** $\approx 117,000\text{ ops/s}$ with mean latency of $7.5\,\mu\text{s}$.
- **20-Branch Dispatch & Collapse:** Sub-$30\text{ms}$ total pipeline duration.
