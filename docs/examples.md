# 📚 Production Cookbooks & Examples

Welcome to the **RiftPoint** cookbooks. These runnable examples illustrate how to build high-performance, multiversal LangGraph agent systems backed by Rust-powered Janus-Tachyon state management.

---

## 🚀 Getting Started

All examples can be executed directly from the repository root using `uv`:

```bash
uv run python examples/01_quickstart.py
```

---

## 📖 Cookbook Index

| Script | Title | Key Concepts |
| :--- | :--- | :--- |
| `01_quickstart.py` | **Minimal LangGraph Quickstart** | `RiftCheckpointSaver`, `RiftRunner`, Branching, Multiverse Collapse, DAG visualization |
| `02_speculative_tool_racing.py` | **Speculative Tool Racing** | `@speculative_node` decorator, concurrent Cache vs RAG vs Web racing, metadata inspection |
| `03_beam_search_reasoning.py` | **Multi-Depth Tree-of-Thought Search** | `BeamSearchRunner`, `BeamSearchConfig`, automatic multi-level branch pruning, convergence |
| `04_time_travel_debugger.py` | **DAG Time Travel & Counterfactual Debugging** | Checkpoint history listing, state rewinding, "what-if" counterfactual forks, Mermaid DAG |
| `05_fastapi_multiverse_agent.py` | **Asynchronous Multiverse Agent Service** | `AsyncRiftCheckpointSaver`, async speculative dispatch, FastAPI endpoint blueprints |

---

## 🛠️ Running the Cookbooks

### 1. Minimal Quickstart

Spawns parallel speculative candidate realities from a canonical checkpoint, evaluates the candidates, and commits the winner:

```bash
uv run python examples/01_quickstart.py
```

### 2. Speculative Tool Racing (`@speculative_node`)

Races multiple tools concurrently (In-Memory Cache vs Vector RAG vs Live Web Search) within a LangGraph node:

```bash
uv run python examples/02_speculative_tool_racing.py
```

### 3. Beam Search Multi-Depth Reasoning (`BeamSearchRunner`)

Orchestrates multi-depth Tree-of-Thought search ($K=2$ beam width, $B=3$ branch factor, $D=4$ depth) with Janus DAG memory pruning:

```bash
uv run python examples/03_beam_search_reasoning.py
```

### 4. Time Travel & Counterfactual Debugging

Demonstrates rewinding an agent to an earlier checkpoint and exploring alternate trajectories:

```bash
uv run python examples/04_time_travel_debugger.py
```

### 5. Async Multiverse Agent Service

Simulates high-throughput asynchronous agent serving with `AsyncRiftCheckpointSaver` and non-blocking multiverse collapse:

```bash
uv run python examples/05_fastapi_multiverse_agent.py
```
