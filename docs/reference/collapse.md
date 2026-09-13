# `riftpoint.collapse`

The `riftpoint.collapse` module provides evaluator protocols and the multiverse resolver for scoring candidate trajectories, committing the winner, and pruning non-winning branches.

---

## Resolver

::: riftpoint.collapse.resolver.MultiverseResolver
    options:
      show_root_heading: true
      members:
        - collapse
        - acollapse

---

## Evaluators

::: riftpoint.collapse.evaluators.HeuristicEvaluator
    options:
      show_root_heading: true

---

::: riftpoint.collapse.evaluators.JSONSchemaEvaluator
    options:
      show_root_heading: true

---

::: riftpoint.collapse.evaluators.LLMJudgeEvaluator
    options:
      show_root_heading: true

---

::: riftpoint.collapse.evaluators.ConsensusEvaluator
    options:
      show_root_heading: true
