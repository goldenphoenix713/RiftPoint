# RiftPoint 🌌

[![CI](https://github.com/goldenphoenix713/RiftPoint/actions/workflows/ci.yml/badge.svg)](https://github.com/goldenphoenix713/RiftPoint/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Core: Tachyon-RS](https://img.shields.io/badge/core-Janus--Tachyon--RS-orange.svg)](https://github.com/goldenphoenix713/janus)

> **High-performance multiversal checkpointer and speculative state engine for LangGraph, powered by Janus-Tachyon-RS.**

---

## 🌟 Overview

In standard LLM agent workflows, exploring alternative reasoning paths, evaluating multiple tool-use strategies, or testing speculative futures requires expensive state copying or complex DAG orchestration.

**RiftPoint** bridges [**LangGraph**](https://github.com/langchain-ai/langgraph) with the Rust-backed [**`janus-tachyon-rs`**](https://github.com/goldenphoenix713/janus) engine to provide:

- **Multiversal Branching:** Instant state forks into candidate realities with near-zero memory duplication penalty.
- **Speculative Tool Calling:** Concurrent execution of divergent tool-use strategies across distinct timeline branches.
- **Multiverse Collapse:** Automatic scoring of candidate timelines (via LLM-as-a-judge, heuristics, or consensus) to commit winning trajectories to `main` while pruning discarded paths.
- **Full State Travel & Tracing:** Interactive time travel, undo/redo, and timeline visualization via Mermaid.js and Matplotlib.

---

## 📚 Planning & Architecture Documentation

Detailed architectural blueprints and implementation phases are documented in [docs/planning](docs/planning/README.md):

- **[Master Architecture & Overview Plan](docs/planning/overview_plan.md)**
- **[Performance Benchmarks & Multiverse Capability Report](docs/BENCHMARKS.md)**
- **[Future Directions & Strategic Roadmap](docs/FUTURE_DIRECTIONS.md)**
- **[Phase 1: Core Janus Bridge & Sync Checkpointer](docs/planning/phase_1_core_bridge_and_sync_checkpointer.md)**
- **[Phase 2: Asynchronous Checkpointer](docs/planning/phase_2_async_checkpointer.md)**
- **[Phase 3: Branching & Speculative Execution Engine](docs/planning/phase_3_branching_and_speculative_runner.md)**
- **[Phase 4: Multiverse Collapse & Evaluation Engine](docs/planning/phase_4_multiverse_collapse_and_evaluation.md)**
- **[Phase 5: Visualization, Serialization & Tracing](docs/planning/phase_5_visualization_and_serialization.md)**

---

## 🚀 Quickstart

### Installation

```bash
pip install riftpoint
```

*Or install with `uv`:*

```bash
uv add riftpoint
```

### Basic Example

```python
from langgraph.graph import StateGraph
from riftpoint import RiftCheckpointSaver

# Initialize the Janus-backed checkpointer
checkpointer = RiftCheckpointSaver()

# Build your LangGraph workflow
builder = StateGraph(...)
# ... define nodes and edges ...
graph = builder.compile(checkpointer=checkpointer)

# Run workflow with multiversal checkpointing
config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke({"messages": [...]}, config=config)

# Render the timeline DAG as a Mermaid diagram
print(checkpointer.visualize("session-1"))
```

---

## 🛠️ Developer Commands

Development workflows use `uv`:

```bash
# Sync dependencies
uv sync --all-groups

# Run tests with coverage requirement (>= 70%)
uv run pytest

# Lint and format
uv run ruff check --fix
uv run ruff format

# Type check
uv run mypy src tests

# Code complexity check (Radon Grade B or better)
uv run python scripts/check_radon.py src
```

---

## 📚 Documentation & Roadmap

- **[Documentation Site](https://riftpoint.readthedocs.io/):** Full guides, API reference, and tutorials.
- **[Roadmap to v1.0](docs/ROADMAP_V1.md):** 5-pillar production readiness checklist toward GA release.
- **[Future Directions](docs/FUTURE_DIRECTIONS.md):** Long-term vision and feature exploration horizon.

---

## 📄 License

This project is dual-licensed under:

- **MIT License** ([LICENSE-MIT](LICENSE-MIT))
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE))
