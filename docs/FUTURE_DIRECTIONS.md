# RiftPoint Future Directions & Architectural Roadmap

This document outlines the strategic engineering roadmap and planned capabilities for **RiftPoint**, extending its core multiversal state engine into high-level agent reasoning, autonomous search, developer ergonomics, and observability.

---

## 🗺️ Roadmap Overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                      RiftPoint Future Architecture                      │
├────────────────────────────────┬────────────────────────────────────────┤
│  Autonomous Tree Search        │  Developer Ergonomics                  │
│  • BeamSearchRunner            │  • @speculative_node Decorator         │
│  • MCTSRunner (UCB1 & Rollout) │  • Middleware Interceptor              │
├────────────────────────────────┼────────────────────────────────────────┤
│  Evaluation & Collapse         │  Observability & DX                    │
│  • Pareto Multi-Objective      │  • TimelineDiff & State Comparator     │
│  • Semantic Consensus / Voting │  • Interactive HTML/SVG DAG Explorer   │
│  • Async LLM-as-a-Judge        │  • astream_multiverse Event Pipeline   │
└────────────────────────────────┴────────────────────────────────────────┘
```

---

## 🌲 1. Autonomous Tree Search Orchestrators

While RiftPoint's [`RiftRunner`](../src/riftpoint/runner/multiverse.py) provides 1-hop parallel branching, complex reasoning workflows (such as mathematical deduction, code generation, and multi-step planning) require multi-level heuristic exploration.

### 1.1 `BeamSearchRunner`

An automated tree-search engine maintaining a fixed beam of the top $K$ most promising candidate trajectories at each depth level:

- **Configurable Parameters:**
  - `beam_width` ($K$): Number of parallel candidate timelines retained per iteration.
  - `branch_factor` ($B$): Number of speculative expansions generated per active beam.
  - `max_depth` ($D$): Maximum reasoning steps before termination.
  - `prune_discarded`: Automatically evicts pruned candidate realities from Janus DAG storage to maintain bounded memory footprint.

```python
from riftpoint import BeamSearchRunner, HeuristicEvaluator

beam_runner = BeamSearchRunner(
    saver=saver,
    beam_width=3,
    branch_factor=4,  # Explores 3 * 4 = 12 universes per step
    max_depth=5,
)

result = await beam_runner.arun(
    graph=reasoning_graph,
    initial_config={"configurable": {"thread_id": "math_problem_1"}},
    expand_fn=generate_candidate_steps,
    evaluator=HeuristicEvaluator(scorer=score_partial_proof),
)

print(f"Optimal Trajectory: {result.winning_trajectory}")
```

### 1.2 `MCTSRunner` (Monte Carlo Tree Search for Agents)

A full MCTS implementation leveraging Janus's sub-millisecond Copy-on-Write (CoW) state rollouts:

1. **Selection:** Traverses existing search tree using Upper Confidence Bound applied to Trees (**UCB1**):
   $$\text{UCB1}(s, a) = Q(s, a) + c \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}$$
2. **Expansion:** Creates candidate multiverse branches via `copy_checkpoint_entry_cross_thread`.
3. **Simulation (Rollout):** Concurrently runs fast speculative policies to terminal evaluation states.
4. **Backpropagation:** Updates reward values and visit counts across the Janus DAG hierarchy.

---

## ⚡ 2. Declarative LangGraph Node Decorators & Middleware

To minimize boilerplate in LangGraph definitions, RiftPoint will provide native functional decorators that turn standard graph nodes into self-collapsing speculative races.

### 2.1 `@speculative_node`

```python
from riftpoint.decorators import speculative_node
from riftpoint.collapse import JSONSchemaEvaluator


@speculative_node(
    strategies=[
        {"name": "fast_cache", "tool": query_redis_cache},
        {"name": "vector_rag", "tool": query_vector_database},
        {"name": "live_web", "tool": query_web_search_api},
    ],
    evaluator=JSONSchemaEvaluator(schema=AnswerSchema),
    prune_discarded=True,
    timeout_seconds=2.0,
)
def search_agent_step(state: AgentState) -> AgentState:
    """Dispatches all 3 strategies concurrently, scores results, and promotes the winner."""
    ...
```

### 2.2 LangGraph Multiverse Middleware

Intercepts state transitions globally to record execution telemetry, track token burn across branches, and provide automatic retry-on-failure by branching from the last successful parent checkpoint.

---

## 🧠 3. Advanced Evaluator Protocols & Consensus Engine

Expanding beyond single-score evaluators to support complex multi-criteria decision making.

### 3.1 `ParetoEvaluator` (Multi-Objective Optimization)

In production agent deployments, the best answer is not merely the highest accuracy score, but an optimal trade-off between **accuracy**, **latency**, and **financial cost**.

- Constructs a Pareto frontier across:
  - $f_1(\text{branch})$: Solution quality score ($\in [0, 10]$)
  - $f_2(\text{branch})$: Execution latency ($\text{ms}$)
  - $f_3(\text{branch})$: LLM token usage / monetary cost ($\$$)
- Automatically filters dominated candidates and selects the knee point of the frontier.

### 3.2 `ConsensusEvaluator` (Majority Voting & Semantic Clustering)

For mission-critical tasks (e.g. medical triage or code verification):

- Concurrently executes $N$ independent reasoning branches (Self-Consistency with CoT).
- Embeds output responses into semantic vector space.
- Clusters answers using DBSCAN / Cosine Similarity to identify majority consensus.
- Selects the canonical representative candidate from the largest cluster.

### 3.3 `AsyncLLMJudgeEvaluator`

- Native non-blocking LLM judge with batched async calls (`ainvoke`).
- Structured rubric parsing (e.g., faithfulness, relevancy, safety).

---

## 🔍 4. Timeline Diffing & Counterfactual Inspector

Provides deep observability into why divergent branches produced differing agent actions.

### 4.1 `TimelineDiff` API

```python
# Diff two branches or historical moments
diff = saver.diff_branches(
    thread_id="customer-support-session",
    branch_a="tool_strategy_rag",
    branch_b="tool_strategy_web",
)

print(diff.render_rich_table())
```

**Output Inspector:**

| Channel / Key | Branch A (`rag`) | Branch B (`web`) | Mutation Type |
| :--- | :--- | :--- | :--- |
| `retrieved_docs` | 3 chunks from Pinecone | 5 articles from Brave API | Divergent Data |
| `tool_calls` | `["query_knowledge_base"]` | `["search_live_web"]` | Different Action |
| `confidence` | `0.81` | `0.94` | Score Delta (+0.13) |
| `latency_ms` | `38.2 ms` | `92.4 ms` | +54.2 ms |

---

## 🌐 5. Interactive HTML/SVG Browser DAG Explorer

Extends the current Mermaid and Matplotlib visualizers with an interactive browser-based visualization tool.

- **Zero-Dependency Standalone HTML:** Outputs self-contained `.html` files using D3.js or Cytoscape.
- **Features:**
  - Pan, zoom, and collapsible branch subtrees.
  - Interactive node click to inspect full state JSON payloads and channel diffs.
  - Visual distinction between the **canonical winning path** (bold green) and **pruned graveyard branches** (dashed gray).
  - Native integration with Jupyter notebooks via `IPython.display.HTML`.

```python
# Generate interactive visualizer
saver.export_interactive_html("session_dag.html", thread_id="agent-run-42")
```

---

## 📡 6. Multiverse Event Streaming (`astream_multiverse`)

Enables real-time streaming of multi-branch agent thought processes directly to user interfaces.

```python
async for event in runner.astream_multiverse(graph, config, branch_specs):
    match event:
        case BranchSpawnedEvent(branch_name, parent_id):
            print(f"🌱 Forked candidate reality: {branch_name}")
        case BranchTokenChunkEvent(branch_name, token):
            print(f"[{branch_name}] {token}", end="")
        case BranchScoredEvent(branch_name, score):
            print(f"📊 {branch_name} evaluated: {score}")
        case MultiverseCollapsedEvent(winner, pruned_branches):
            print(f"🏆 Reality collapsed! Winner: {winner}, Pruned: {pruned_branches}")
```

---

## 📊 Summary of Strategic Impact

| Feature Area | User & Developer Value | Architectural Enabler |
| :--- | :--- | :--- |
| **Tree Search (`BeamSearch` / `MCTS`)** | Multi-step agent planning & complex problem solving | Janus $O(1)$ CoW branching |
| **`@speculative_node` Decorator** | Zero boilerplate integration into existing LangGraph apps | `RiftRunner` + `MultiverseResolver` |
| **`ParetoEvaluator` & `Consensus`** | Enterprise-grade reliability & cost/latency trade-offs | Dynamic scoring protocol |
| **`TimelineDiff` Inspector** | Counterfactual debugging & agent auditing | Janus moment checkpoint index |
| **Interactive HTML DAG** | Real-time visual timeline debugging in browsers | Standalone D3/SVG exporter |
| **Multiverse Streaming** | Low-latency UI streaming for parallel agent reasoning | Async event emitter pipeline |
