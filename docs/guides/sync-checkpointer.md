# Synchronous Checkpointing

The [`RiftCheckpointSaver`](../reference/checkpointer.md) provides synchronous, thread-safe multiversal checkpointing for LangGraph workflows.

---

## 🔧 Basic Setup

```python
from langgraph.graph import StateGraph
from riftpoint import RiftCheckpointSaver

saver = RiftCheckpointSaver()
graph = builder.compile(checkpointer=saver)
```

---

## 🔒 Concurrency Control & Thread Safety

`RiftCheckpointSaver` acquires fine-grained per-thread locks during write and read operations. When multiple threads execute branches concurrently in a `ThreadPoolExecutor`:

- Reads and writes to different `thread_id`s operate completely in parallel without blocking.
- Writes to the same `thread_id` are serialized safely to prevent corrupted DAG lineages.

```python
# Multi-threaded execution is 100% thread-safe
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = [
        pool.submit(
            graph.invoke, {"step": i}, {"configurable": {"thread_id": f"thread_{i}"}}
        )
        for i in range(8)
    ]
```

---

## 🔍 Reading Checkpoints & State Tuples

```python
config = {"configurable": {"thread_id": "session-1"}}

# Get current latest state tuple
state_tuple = saver.get_tuple(config)
if state_tuple:
    print("Current Checkpoint ID:", state_tuple.checkpoint["id"])
    print("Channel Values:", state_tuple.checkpoint["channel_values"])

# List historical checkpoints for thread
for chk_tuple in saver.list(config, limit=5):
    print(f"Historical Step: {chk_tuple.checkpoint['id']}")
```

---

## 🌿 Spawning Branches Synchronously

```python
# Fork a branch directly in storage
saver.copy_checkpoint_entry_cross_thread(
    from_thread_id="session-1",
    from_namespace="",
    to_thread_id="session-1:candidate_branch",
    to_namespace="",
    checkpoint_id=state_tuple.checkpoint["id"],
)
```
