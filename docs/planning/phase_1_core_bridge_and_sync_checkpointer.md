# Phase 1: Core Janus Bridge & Sync Checkpointer

## 1. Objective & Scope

Establish the fundamental integration layer between **LangGraph** and the Rust-backed **`janus-tachyon-rs`** engine. Deliver the synchronous `RiftCheckpointSaver` subclassing LangGraph's `BaseCheckpointSaver`.

---

## 2. Key Technical Specifications

### 2.1 Interface & Protocol Mapping

`RiftCheckpointSaver` implements the standard LangGraph `BaseCheckpointSaver` interface:

- **`get_tuple(config: RunnableConfig) -> Optional[CheckpointTuple]`**
  - Extracts `thread_id` and `checkpoint_id` (or defaults to the latest head).
  - Retrieves the corresponding state snapshot from Janus Tachyon DAG.
  - Returns a populated `CheckpointTuple` with config, checkpoint state, metadata, parent config, and pending writes.

- **`list(config: Optional[RunnableConfig], *, filter: Optional[dict] = None, before: Optional[RunnableConfig] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]`**
  - Traverses the checkpoint history for a given `thread_id` across branches.
  - Supports filtering by metadata and limiting result count.

- **`put(config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig`**
  - Stores a new checkpoint state into the Janus timeline for the active `thread_id`.
  - Automatically handles parent-child branch points when `checkpoint_id` differs from the current branch head.
  - Returns updated `RunnableConfig` containing `thread_id`, `checkpoint_ns`, and new `checkpoint_id`.

- **`put_writes(config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str) -> None`**
  - Stores intermediate channel writes associated with a task execution step.

### 2.2 Janus State Representation

- Each `thread_id` maintains an instance of `janus.MultiverseBase`.
- Checkpoints are saved as state commits in the multiverse DAG.
- Diffing and branch points are managed with zero-copy semantics inside the Rust core.

---

## 3. Module & File Deliverables

- `src/riftpoint/checkpointer/__init__.py`: Checkpointer exports.
- `src/riftpoint/checkpointer/base.py`: Core bridge abstractions and thread-multiverse registry.
- `src/riftpoint/checkpointer/sync_saver.py`: `RiftCheckpointSaver` implementation.
- `tests/test_sync_checkpointer.py`: Unit tests validating `get_tuple`, `list`, `put`, `put_writes`, and branch points.

---

## 4. Verification & Quality Criteria

1. **Strict Mypy Type Checking:** Full type coverage with zero untyped definitions (`strict = true`).
2. **Radon Metric Enforcement:** CC $\le 10$ and MI $\ge 10$ (Grade B or better).
3. **Test Coverage:** $\ge 70\%$ test coverage across all new modules.
4. **LangGraph Compatibility:** Successful execution of a standard compiled `StateGraph` using `RiftCheckpointSaver`.
