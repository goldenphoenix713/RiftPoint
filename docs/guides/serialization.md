# Snapshot Serialization & SerDe

RiftPoint supports exporting and restoring full multiversal session snapshots in multiple serialization formats (JSON, MessagePack, and binary).

---

## 💾 Exporting Session Snapshots

Export an entire session's DAG moments and lineage to a JSON string or bytes:

```python
# Export session to JSON string
json_snapshot = saver.export_session_json("session-1")

with open("session_1_backup.json", "w") as f:
    f.write(json_snapshot)
```

---

## 🔄 Restoring Session Snapshots

Import a previously saved snapshot into a new checkpointer instance:

```python
new_saver = RiftCheckpointSaver()

with open("session_1_backup.json", "r") as f:
    json_snapshot = f.read()

new_saver.import_session_json(json_snapshot, target_thread_id="restored-session-1")

# Verify restored checkpoint state
config = {"configurable": {"thread_id": "restored-session-1"}}
restored_tuple = new_saver.get_tuple(config)
print("Restored state:", restored_tuple.checkpoint["channel_values"])
```
