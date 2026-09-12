# RiftPoint Planning & Architecture Documentation

Welcome to the planning and architecture hub for **RiftPoint**, the multiversal checkpointer for LangGraph powered by Janus-Tachyon-RS.

---

## 🗺️ Master Architecture & Implementation Plan

- **[High-Level Architecture & Overview Plan](file:///Users/eduardo.ruiz/PycharmProjects/RiftPoint/docs/planning/overview_plan.md):** Core mission, system design decisions, module layout, and high-level milestones.

---

## 📑 Phased Implementation Plans

1. **[Phase 1: Core Janus Bridge & Sync Checkpointer](file:///Users/eduardo.ruiz/PycharmProjects/RiftPoint/docs/planning/phase_1_core_bridge_and_sync_checkpointer.md)**
   - Integration with `janus-tachyon-rs`.
   - Implementation of `RiftCheckpointSaver` (`BaseCheckpointSaver`).
   - Per-thread multiversal state mapping, commit history, and basic branching.

2. **[Phase 2: Asynchronous Checkpointer](file:///Users/eduardo.ruiz/PycharmProjects/RiftPoint/docs/planning/phase_2_async_checkpointer.md)**
   - Implementation of `AsyncRiftCheckpointSaver`.
   - Non-blocking async queries, concurrency control, and thread-safe locks.

3. **[Phase 3: Branching & Speculative Execution Engine](file:///Users/eduardo.ruiz/PycharmProjects/RiftPoint/docs/planning/phase_3_branching_and_speculative_runner.md)**
   - `BranchManager` for creating and switching timeline branches.
   - `RiftRunner` for concurrent execution of speculative agent reasoning and tool calls.

4. **[Phase 4: Multiverse Collapse & Evaluation Engine](file:///Users/eduardo.ruiz/PycharmProjects/RiftPoint/docs/planning/phase_4_multiverse_collapse_and_evaluation.md)**
   - Scoring protocols and built-in evaluators (LLM judge, heuristic rules, consensus).
   - `MultiverseResolver` to merge the winning timeline and prune discarded branches.

5. **[Phase 5: Visualization, Serialization & Tracing](file:///Users/eduardo.ruiz/PycharmProjects/RiftPoint/docs/planning/phase_5_visualization_and_serialization.md)**
   - Mermaid diagram generation and Matplotlib DAG rendering.
   - JSON, MsgPack, and binary state export/import.
