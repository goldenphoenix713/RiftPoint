# Multiverse Collapse & Evaluators

The collapse phase in RiftPoint evaluates multiple candidate branch outputs, assigns quality scores, commits the winning state to `main`, and evicts discarded branches from Janus storage.

---

## 🎯 Evaluator Protocols

RiftPoint includes several built-in evaluator protocols:

### 1. `HeuristicEvaluator`

Evaluates branch results using a custom deterministic scoring callable:

```python
from riftpoint import HeuristicEvaluator

evaluator = HeuristicEvaluator(
    scorer=lambda result: float(
        result.output.get("confidence", 0.0) if result.is_success else 0.0
    )
)
```

### 2. `JSONSchemaEvaluator`

Validates candidate outputs against a JSON Schema or Pydantic model:

```python
from riftpoint import JSONSchemaEvaluator

schema = {
    "type": "object",
    "required": ["answer", "citations"],
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
}

evaluator = JSONSchemaEvaluator(schema=schema)
```

### 3. `LLMJudgeEvaluator`

Scores candidates using an LLM evaluator function:

```python
from riftpoint import LLMJudgeEvaluator


def judge_candidate(result) -> float:
    # Call OpenAI / Anthropic / Local LLM
    return 8.5


evaluator = LLMJudgeEvaluator(judge_fn=judge_candidate)
```

---

## 🧹 Automatic Branch Pruning

Setting `prune_discarded=True` in `collapse()` or `acollapse()` instructs RiftPoint to delete all non-winning candidate branch checkpoints from storage:

```python
collapse_result = resolver.collapse(
    thread_id="session-1",
    results=results,
    evaluator=evaluator,
    prune_discarded=True,  # Cleans up 100% of discarded branch allocations
)
```

This prevents memory growth in long-running agent workflows.
