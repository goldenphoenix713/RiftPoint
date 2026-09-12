# Phase 4: Multiverse Collapse & Evaluation Engine

## 1. Objective & Scope

Implement the evaluation, scoring, and collapse framework that evaluates candidate realities, selects the optimal trajectory, commits the winner back to the canonical branch (`main`), and prunes or archives discarded branches.

---

## 2. Key Technical Specifications

### 2.1 Evaluators & Scoring Protocols

- **`BaseBranchEvaluator` (Protocol / Abstract Base Class):**
  - Defines the standard signature for scoring candidate branch outcomes:
    `evaluate(branch_id: str, state: Dict[str, Any], metadata: Dict[str, Any]) -> EvaluationScore`

- **Built-in Evaluators:**
  - **`HeuristicEvaluator`:** Rule-based verification (schema conformance, execution success, regex matching, cost/token metrics).
  - **`LLMJudgeEvaluator`:** Evaluates reasoning quality and task completion using structured LLM outputs.
  - **`ConsensusEvaluator`:** Calculates consensus and agreement scores across multi-branch runs.

### 2.2 Collapse Engine (`MultiverseResolver`)

- **`collapse_multiverse(thread_id: str, results: Dict[str, BranchResult], evaluator: BaseBranchEvaluator, target_branch: str = "main", prune_discarded: bool = True) -> CollapseResult`**
  - Scores all candidate realities against the evaluation criteria.
  - Identifies the winning trajectory.
  - Merges the winning state into `target_branch` via Janus Tachyon container-aware merging.
  - Prunes discarded candidate branches to reclaim memory.

---

## 3. Module & File Deliverables

- `src/riftpoint/collapse/__init__.py`: Collapse and evaluator exports.
- `src/riftpoint/collapse/evaluators.py`: Protocol definition and built-in evaluators.
- `src/riftpoint/collapse/resolver.py`: `MultiverseResolver` and collapse logic.
- `tests/test_collapse_resolver.py`: Tests for scoring, selection, merging, and branch pruning.

---

## 4. Verification & Quality Criteria

1. **Deterministic Resolution:** Predictable winner selection based on evaluator scores.
2. **Clean Pruning:** Discarded candidate nodes are properly removed without corrupting the DAG.
3. **Mypy Strict & Radon Metrics:** Zero untyped definitions, Grade B or better on CC and MI.
4. **Test Coverage:** $\ge 70\%$ test coverage on collapse modules.
