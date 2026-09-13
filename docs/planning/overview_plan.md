# RiftPoint Architecture & Implementation Plan

## 1. Executive Summary

**RiftPoint** is a high-performance, multiversal checkpointer and state branching engine designed for [LangGraph](https://github.com/langchain-ai/langgraph). Powered by the Rust-backed [**`janus-tachyon-rs`**](https://github.com/goldenphoenix713/janus) (`Tachyon-RS`) core, RiftPoint transforms agent execution from linear graphs into branching multiverses where agents can speculatively test hypotheses, explore divergent tool strategies, score candidate trajectories, and collapse to the optimal reality with near-zero memory duplication penalty.

---

## 2. Core Architecture & Design Decisions

### 2.1 Engine Binding & State Isolation

- **Per-Session Multiverse Instance:** Each primary LangGraph session (`thread_id`) manages an isolated `janus.MultiverseBase` instance stored in memory.
- **Intra-Multiverse Speculative Branching:** Speculative workers and candidate timelines fork as branches *inside* that session's `MultiverseBase` DAG rather than spawning separate multiverses. This allows workers to share historical checkpoints with zero memory duplication.
- **Tachyon-RS Zero-Copy Diffing:** States are recorded as structural diffs inside Tachyon-RS, preventing full memory duplication when branching into dozens of speculative timelines.
- **Export & Import:** In-memory multiverses support export/import to JSON, MsgPack, and binary streams for persistence and distributed debugging.

### 2.2 Checkpointer Layer (`RiftCheckpointSaver` & `AsyncRiftCheckpointSaver`)

- **Dual Sync/Async Support:** Full implementation of both `BaseCheckpointSaver` and `AsyncCheckpointSaver` protocols from `langgraph.checkpoint.base`.
- **Checkpoint Mapping:**
  - `thread_id`: Maps to a specific multiverse DAG in Janus (or a named branch within that multiverse).
  - `checkpoint_id`: Maps to a specific node/commit in the Janus multiverse tree.
  - `parent_config`: Tracks ancestry for branching and time travel.
  - `checkpoint_ns`: Namespaces subgraphs or isolated worker branch trajectories within the same multiverse.
  - `pending_writes`: Intermediate node writes are buffered per checkpoint.

### 2.3 Branching & Speculative Runner (`RiftRunner`)

- **Multiversal Branching:** High-level orchestration helpers allowing developers to fork execution from any checkpoint into $N$ candidate timelines.
- **Speculative Tool Calling:** Concurrent execution of divergent tool-use strategies across isolated branches without state collisions.
- **Dynamic Threading:** Parent-to-child timeline spawning where child branches inherit exact snapshots with delta tracking.

### 2.4 Multiverse Collapse & Resolution Framework

- **Flexible Evaluators:** Built-in support for:
  - **LLM-as-a-Judge:** Automated evaluation prompts scoring outcomes across branches.
  - **Heuristic & Rule-Based Scorers:** Unit test assertions, JSON schema validators, or reward functions.
  - **Consensus Voting:** Multi-agent agreement metrics across divergent runs.
- **Collapse Mechanics:** Winning branch trajectory commits to the canonical timeline (`main`), while discarded candidate branches are safely pruned or archived.

### 2.5 Visualization & Tracing

- **Direct Checkpointer Methods:**
  - `saver.visualize(thread_id)`: Outputs rich Mermaid.js syntax for inline markdown rendering or web dashboards.
  - `saver.plot(thread_id, backend="matplotlib")`: Generates graphical DAG plots via Matplotlib and NetworkX.
- **Tracing & Time-Travel:** Step backward and forward across checkpoints, diff alternate realities, and inspect intermediate channel states.

---

## 3. Repository & Module Layout

```text
src/riftpoint/
├── __init__.py                # Public API exports
├── py.typed                   # PEP 561 typing marker
├── checkpointer/
│   ├── __init__.py            # Checkpointer exports
│   ├── base.py                # Core checkpointer logic & Janus bridge
│   ├── sync_saver.py          # RiftCheckpointSaver (Sync)
│   └── async_saver.py         # AsyncRiftCheckpointSaver (Async)
├── runner/
│   ├── __init__.py            # Runner exports
│   ├── branch.py              # Branch creation & management
│   └── multiverse.py          # RiftRunner for speculative execution
├── collapse/
│   ├── __init__.py            # Collapse framework exports
│   ├── evaluators.py          # Built-in scoring functions (LLM judge, heuristic)
│   └── resolver.py            # Multiverse collapse and branch commit logic
├── visualization/
│   ├── __init__.py            # Visualization exports
│   ├── mermaid.py             # Mermaid DAG generator
│   └── plot.py                # Matplotlib / NetworkX renderer
└── serde/
    ├── __init__.py            # Serialization exports
    └── codec.py               # JSON, MsgPack, and binary export/import
```

---

## 4. Phased Implementation Roadmap

### Phase 1: Core Janus Bridge & Synchronous Checkpointer

- Implement `RiftCheckpointSaver` subclassing `BaseCheckpointSaver`.
- Wire `get_tuple`, `list`, `put`, `put_writes` to per-thread `janus.MultiverseBase`.
- Add unit tests verifying state storage, versioning, and parent-child linkage.

### Phase 2: Asynchronous Checkpointer (`AsyncRiftCheckpointSaver`)

- Implement `AsyncRiftCheckpointSaver` subclassing `AsyncCheckpointSaver`.
- Wrap Janus operations with non-blocking async execution patterns.
- Add async pytest suites testing concurrent state reads/writes.

### Phase 3: Branching & Speculative Execution Engine (`RiftRunner`)

- Build high-level helpers for branching graphs into candidate realities.
- Support parallel subgraph execution across divergent branches.
- Implement branch isolation guarantees and rollback safeguards.

### Phase 4: Multiverse Collapse & Evaluation Suite

- Implement scoring protocols and built-in evaluators (LLM judge, heuristic rules).
- Build the `collapse_multiverse` engine to commit winner states and prune alternate realities.
- Provide end-to-end examples demonstrating speculative reasoning workflows.

### Phase 5: Visualization, Serialization & Tracing

- Implement `visualize(thread_id)` for Mermaid diagram generation.
- Implement `plot(thread_id)` with Matplotlib backend.
- Build serialization codecs for state export and import.
- Complete documentation, integration guides, and end-to-end examples.

---

## 5. Quality & Verification Standards

- **Strict Type Checking:** `mypy` strict mode with zero untyped definitions across all modules.
- **Code Complexity:** Enforce Radon Grade B or better ($CC \le 10$, $MI \ge 10$) via `scripts/check_radon.py`.
- **Test Coverage:** Enforce $\ge 70\%$ coverage on every module using `pytest` + `pytest-cov`.
- **Code Style:** Strict Ruff configuration (88 character line limit, all rulesets enabled).
