# RiftPoint v1.0.0 Production Roadmap

This document defines the official engineering roadmap and readiness checklist to advance **RiftPoint** from its current **`v0.9.0` (Feature-Complete Release Candidate)** state to **`v1.0.0` (General Availability / Production Stable)**.

---

## 🎯 Current Maturity & v1.0 Goal

| Dimension | Current State (`v0.9.0`) | Target State (`v1.0.0`) |
| :--- | :--- | :--- |
| **PyPI Status** | `Development Status :: 3 - Alpha` | `Development Status :: 5 - Production/Stable` |
| **Core Checkpointing** | Sync & Async Janus-backed savers | Hardened against all LangGraph edge cases |
| **Search Engine** | 1-hop parallel branching (`RiftRunner`) | Autonomous `BeamSearchRunner` & MCTS engine |
| **Developer DX** | Manual runner & resolver composition | Declarative `@speculative_node` decorator |
| **Code Coverage** | 88.95% (50 unit tests) | **$\ge 95.0\%$** across all edge cases |
| **Cookbooks & Demos** | Basic quickstarts & benchmarks | 5 production runnable scripts in `examples/` |
| **CI/CD & Release** | Local test runners & GitHub Actions CI | Multi-OS matrix + Automated PyPI publishing |

---

## 🗺️ The Five Pillars to v1.0.0

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        RiftPoint v1.0.0 Pillars                        │
├───────────────────────────────┬────────────────────────────────────────┤
│  1. Autonomous Tree Search    │  2. Declarative Ergonomics             │
│  • BeamSearchRunner           │  • @speculative_node Decorator         │
│  • Multi-Level Auto-Pruning   │  • LangGraph Middleware Wrapper        │
├───────────────────────────────┼────────────────────────────────────────┤
│  3. Production Cookbooks      │  4. Quality Hardening (≥ 95%)          │
│  • 5 Runnable Example Scripts │  • Error Recovery & Fault Tolerance    │
│  • FastAPI Multiverse Agent   │  • Snapshot Corruption Hardening       │
├───────────────────────────────┴────────────────────────────────────────┤
│  5. CI/CD & Automated PyPI Release Pipeline                            │
│  • Trusted Publishing (OIDC) via GitHub Actions                        │
│  • Multi-OS Matrix Testing (Ubuntu, macOS, Windows)                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Detailed Pillar Specifications

### 1. Autonomous Tree Search Orchestration (`BeamSearchRunner`)

- **Objective:** Enable multi-step Tree-of-Thought and Beam Search reasoning in a single line of code.
- **Key Components:**
  - **`BeamSearchRunner`:**
    - `beam_width` ($K$): Number of parallel survivor timelines maintained per depth level.
    - `branch_factor` ($B$): Number of candidate thoughts/actions expanded per beam.
    - `max_depth` ($D$): Maximum exploration depth before terminating.
  - **Janus Memory Integration:** Automatically evicts non-surviving branches from Janus DAG memory at each depth boundary to maintain a flat memory footprint.

```python
from riftpoint import BeamSearchRunner, HeuristicEvaluator

beam_runner = BeamSearchRunner(saver=saver, beam_width=3, branch_factor=4, max_depth=5)
result = await beam_runner.arun(
    graph=graph,
    initial_config=root_config,
    expand_fn=generate_candidate_steps,
    evaluator=evaluator,
)
```

---

### 2. Declarative Developer Ergonomics (`@speculative_node`)

- **Objective:** Allow developers to turn any LangGraph node into a speculative race without manual orchestration boilerplate.
- **Key Components:**
  - **`@speculative_node` Decorator:**
    - Takes a list of candidate tool functions or branch specifications.
    - Dispatches all candidates concurrently via `RiftRunner`.
    - Evaluates outputs with a configured evaluator (`JSONSchema`, `Heuristic`, or `LLMJudge`).
    - Automatically collapses to the winner and returns the updated node state.

```python
@speculative_node(
    tools=[fetch_redis_cache, fetch_vector_rag, fetch_live_web],
    evaluator=JSONSchemaEvaluator(schema=AnswerSchema),
    prune_discarded=True,
)
def research_step(state: AgentState) -> AgentState:
    """Automatically races tools and promotes the winner."""
    ...
```

---

### 3. Production Cookbooks & Examples (`examples/`)

- **Objective:** Provide developer-facing, runnable reference implementations.
- **Deliverables:**
  - `examples/01_quickstart.py`: Minimal 20-line LangGraph integration.
  - `examples/02_speculative_tool_racing.py`: Multi-strategy RAG vs SQL vs Web race.
  - `examples/03_beam_search_reasoning.py`: Complex problem solving with beam search.
  - `examples/04_time_travel_debugger.py`: Rewinding state and exploring counterfactuals.
  - `examples/05_fastapi_multiverse_agent.py`: High-throughput async agent serving via FastAPI.

---

### 4. Test Suite Hardening ($\ge 95\%$ Coverage)

- **Objective:** Elevate test coverage from 88.95% to $\ge 95.0\%$ with dedicated failure and edge-case verification.
- **Target Scenarios:**
  - **Branch Fault Isolation:** Verify unhandled exceptions in one candidate branch never crash parallel sibling branches.
  - **Serialization Resilience:** Corrupted or truncated snapshot recovery testing.
  - **Namespace Boundary Isolation:** Deep validation of multi-tenant subgraphs and namespaces.
  - **Thread Lock Stress:** High-load lock acquisition and release under thread starvation conditions.

---

### 5. CI/CD & Automated PyPI Release Pipeline

- **Objective:** Fully automated, tamper-proof release process.
- **Key Components:**
  - **GitHub Actions (`.github/workflows/publish.yml`):**
    - Automated build and upload to PyPI using PyPI Trusted Publishing (OIDC).
    - Triggered exclusively on signed git tags (`v*.*.*`).
  - **Multi-OS CI Matrix:**
    - Automated test suite execution across Ubuntu, macOS (x86_64 and ARM64), and Windows.
    - Matrix validation on Python 3.11, 3.12, and 3.13.
  - **Project Metadata Upgrade:**
    - Set `version = "1.0.0"` in `pyproject.toml`.
    - Upgrade classifier to `Development Status :: 5 - Production/Stable`.

---

## 🗓️ v1.0.0 Execution Timeline

- [x] **Phase 1:** Core Additions (`BeamSearchRunner` & `@speculative_node`)
- [x] **Phase 2:** Production Cookbooks (`examples/` directory)
- [x] **Phase 3:** Edge-Case Hardening & Coverage Boost ($\ge 95\%$)
- [x] **Phase 4:** GitHub Actions PyPI Publishing & Multi-OS CI Workflow
- [ ] **Phase 5:** Formal `v1.0.0` Release Tag & PyPI Publication
