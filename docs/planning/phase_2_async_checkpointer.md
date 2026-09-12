# Phase 2: Asynchronous Checkpointer

## 1. Objective & Scope

Implement `AsyncRiftCheckpointSaver`, providing first-class asynchronous checkpointing capabilities for modern async LangGraph agent graphs and concurrent LLM pipelines.

---

## 2. Key Technical Specifications

### 2.1 Interface & Protocol Mapping

`AsyncRiftCheckpointSaver` subclasses `BaseCheckpointSaver` (or `AsyncCheckpointSaver` from `langgraph.checkpoint.base`):

- **`aget_tuple(config: RunnableConfig) -> Optional[CheckpointTuple]`**
  - Asynchronously retrieves state tuple for the specified `thread_id` and `checkpoint_id`.
  - Dispatches non-blocking queries to the underlying Janus Tachyon state registry.

- **`alist(config: Optional[RunnableConfig], *, filter: Optional[dict] = None, before: Optional[RunnableConfig] = None, limit: Optional[int] = None) -> AsyncIterator[CheckpointTuple]`**
  - Asynchronously yields checkpoint tuples across historical timeline nodes.

- **`aput(config: RunnableConfig, checkpoint: Checkpoint, metadata: CheckpointMetadata, new_versions: ChannelVersions) -> RunnableConfig`**
  - Asynchronously saves checkpoint commits into the thread's multiverse tree.

- **`aput_writes(config: RunnableConfig, writes: Sequence[Tuple[str, Any]], task_id: str) -> None`**
  - Asynchronously persists pending channel writes.

### 2.2 Concurrency & Thread-Safety

- Utilize thread-safe locks (e.g., `asyncio.Lock` or `threading.RLock`) per thread to ensure safe concurrent access from parallel async nodes.
- Maintain isolation between distinct `thread_id` multiverses under heavy async workloads.

---

## 3. Module & File Deliverables

- `src/riftpoint/checkpointer/async_saver.py`: `AsyncRiftCheckpointSaver` implementation.
- `src/riftpoint/checkpointer/__init__.py`: Update exports to include `AsyncRiftCheckpointSaver`.
- `tests/test_async_checkpointer.py`: Comprehensive async pytest suite covering `ainvoke`, `astream`, concurrent `aput` operations, and state isolation.

---

## 4. Verification & Quality Criteria

1. **Async Execution:** Full compatibility with LangGraph's async execution engine (`graph.ainvoke()`, `graph.astream()`).
2. **Strict Mypy Type Checking:** Strict compliance with `AsyncIterator` and typing protocols.
3. **Radon Complexity:** CC $\le 10$ and MI $\ge 10$ (Grade B or better).
4. **Test Coverage:** $\ge 70\%$ test coverage on async modules.
