# DAG Time Travel & Replay

RiftPoint maintains a full immutable lineage graph of all checkpoints in the Rust Janus core, enabling instant historical rewind and counterfactual reasoning.

---

## ⏳ Rewinding to Historical Checkpoints

You can retrieve any prior checkpoint and resume or fork from that exact moment in time:

```python
config = {"configurable": {"thread_id": "session-42"}}

# 1. Inspect checkpoint history
history = list(saver.list(config, limit=10))
for chk_tuple in history:
    print(f"Step {chk_tuple.metadata.get('step')}: {chk_tuple.checkpoint['id']}")

# 2. Target historical step 3
step_3_checkpoint_id = history[2].checkpoint["id"]

# 3. Rewind state to Step 3
rewind_config = {
    "configurable": {
        "thread_id": "session-42",
        "checkpoint_id": step_3_checkpoint_id,
    }
}
historical_tuple = saver.get_tuple(rewind_config)
print("State at Step 3:", historical_tuple.checkpoint["channel_values"])
```

---

## 🔀 Counterfactual Branching from the Past

Spawn alternative realities branching off an earlier historical checkpoint:

```python
# Fork an alternative reality starting from historical Step 3
saver.copy_checkpoint_entry_cross_thread(
    from_thread_id="session-42",
    from_namespace="",
    to_thread_id="session-42:counterfactual_branch",
    to_namespace="",
    checkpoint_id=step_3_checkpoint_id,
)

# Invoke graph on counterfactual branch
cf_config = {"configurable": {"thread_id": "session-42:counterfactual_branch"}}
cf_result = graph.invoke({"alternative_input": True}, config=cf_config)
```
