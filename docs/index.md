# RiftPoint 🌌

> **High-performance multiversal checkpointer and speculative state management engine for LangGraph, powered by Janus-Tachyon-RS.**

---

## 🌟 What is RiftPoint?

In standard LLM agent frameworks, exploring alternate reasoning branches, executing speculative tool strategies, or backtracking requires either expensive state duplication (`copy.deepcopy`) or complex ad-hoc tree orchestration.

**RiftPoint** intercepts LangGraph's execution DAG to enable:

- **⚡ Instant Zero-Copy Branching:** Fork agent state into candidate realities with sub-millisecond $O(1)$ Copy-on-Write (CoW) shared storage—up to **$965\times$ faster** than standard deepcopy.
- **🏎️ Speculative Tool Racing:** Concurrently execute divergent tool strategies across candidate timelines, collapsing to the winning result in sub-100ms.
- **🏆 Automated Multiverse Collapse:** Score candidate universes using structured evaluators (`JSONSchema`, `Heuristic`, `LLMJudge`), promote the winning trajectory to canonical `main`, and automatically evict discarded branches.
- **⏳ DAG Time-Travel & Historical Rewind:** Rewind agent execution to any historical decision point and spawn counterfactual realities without state corruption.
- **📊 Multiverse Observability:** Native timeline visualizations rendered with Mermaid.js diagrams and Matplotlib DAG trees.

---

## 🚀 Quick Look

```python
from langgraph.graph import StateGraph
from riftpoint import (
    BranchSpec,
    HeuristicEvaluator,
    MultiverseResolver,
    RiftCheckpointSaver,
    RiftRunner,
)

# 1. Initialize Janus-backed checkpointer
saver = RiftCheckpointSaver()
runner = RiftRunner(saver=saver)
resolver = MultiverseResolver(saver=saver)

# 2. Build your LangGraph workflow
graph = builder.compile(checkpointer=saver)
root_cfg = {"configurable": {"thread_id": "agent-session"}}

# 3. Concurrently explore 3 candidate tool strategies
results = runner.run_parallel_branches(
    graph=graph,
    initial_config=root_cfg,
    branch_specs=[
        BranchSpec(name="fast_cache", input_data={"query": "cached_val"}),
        BranchSpec(name="vector_rag", input_data={"query": "rag_docs"}),
        BranchSpec(name="live_search", input_data={"query": "web_api"}),
    ],
)

# 4. Collapse candidate realities and auto-prune discarded branches
winner = resolver.collapse(
    thread_id="agent-session",
    results=results,
    evaluator=HeuristicEvaluator(scorer=lambda r: r.output.get("score", 0.0)),
    prune_discarded=True,
)
```

---

## 🧭 Navigation Guide

- **[Getting Started](getting-started/installation.md):** Installation, 5-minute tutorial, and core architectural concepts.
- **[User Guides](guides/sync-checkpointer.md):** Step-by-step guides for sync/async checkpointers, speculative tool racing, multiverse collapse, and time-travel.
- **[Performance Benchmarks](BENCHMARKS.md):** Complete microsecond latency percentiles, memory profiling, and speedup charts.
- **[API Reference](reference/checkpointer.md):** Auto-generated API reference for all classes and functions.
- **[Roadmap & Future Directions](FUTURE_DIRECTIONS.md):** Autonomous tree search (MCTS / Beam search), decorators, and streaming.
