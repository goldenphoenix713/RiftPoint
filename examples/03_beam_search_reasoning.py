"""03_beam_search_reasoning.py — Autonomous Multi-Depth Tree-of-Thought Search.

Demonstrates:
1. Configuring the BeamSearchRunner for multi-depth exploration.
2. Formulating multi-branch candidate expansions at each depth step.
3. Automatically pruning discarded branches from Janus DAG memory.
4. Inspecting per-depth exploration metrics and convergence.
5. Emitting the optimal reasoning trajectory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import (
    BeamSearchConfig,
    BeamSearchRunner,
    BranchSpec,
    HeuristicEvaluator,
    RiftCheckpointSaver,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from riftpoint.runner.multiverse import BranchResult


# 1. State Schema for Multi-Step Reasoning
class ReasoningState(TypedDict):
    """Agent state for step-by-step problem solving."""

    messages: Annotated[list[Any], add_messages]
    problem: str
    steps: list[str]
    current_estimate: float
    target_value: float


# 2. Graph Node: Applies a reasoning step
def apply_step_node(state: ReasoningState) -> dict[str, Any]:
    """Execute a candidate calculation step."""
    steps = list(state.get("steps", []))
    last_step = steps[-1] if steps else "Start"
    return {
        "messages": [AIMessage(content=f"Evaluated step: {last_step}")],
    }


# 3. Expansion Function: Generates candidate arithmetic operations toward target
def expand_reasoning_candidates(
    parent_result: BranchResult,
) -> list[BranchSpec]:
    """Expand 3 candidate operations per beam."""
    state = parent_result.output or {}
    curr = float(state.get("current_estimate", 1.0))
    target = float(state.get("target_value", 42.0))
    existing_steps = list(state.get("steps", []))

    # Candidate operations: multiply, add, exponent/divide
    candidates = [
        (f"*2 (val={curr * 2:.1f})", curr * 2),
        (f"+7 (val={curr + 7:.1f})", curr + 7),
        (f"+15 (val={curr + 15:.1f})", curr + 15),
    ]

    specs: list[BranchSpec] = []
    for idx, (label, new_val) in enumerate(candidates):
        specs.append(
            BranchSpec(
                name=f"op_{idx}",
                input_data={
                    "steps": [*existing_steps, f"Step: {label}"],
                    "current_estimate": new_val,
                    "target_value": target,
                },
            )
        )
    return specs


# 4. Scorer: Proximity to target value (42.0)
def score_proximity(result: BranchResult) -> float:
    """Score candidate by closeness to target value (higher is better)."""
    if not result.is_success or not result.output:
        return 0.0
    val = float(result.output.get("current_estimate", 0.0))
    target = float(result.output.get("target_value", 42.0))
    error = abs(target - val)
    # Inverse distance score in (0, 1]
    return 1.0 / (1.0 + error)


def main() -> None:
    """Run beam search reasoning demonstration."""
    print("=" * 75)
    print("🌳 RiftPoint: Multi-Depth Beam Search Tree-of-Thought Engine")
    print("=" * 75)

    # 5. Build and Compile Graph
    builder = StateGraph(ReasoningState)
    builder.add_node("apply_step", apply_step_node)
    builder.add_edge(START, "apply_step")

    saver = RiftCheckpointSaver()
    graph = builder.compile(checkpointer=saver)

    # 6. Initialize Graph with Root Problem
    thread_id = "beam-search-session"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    target = 42.0
    initial_input: ReasoningState = {
        "messages": [
            HumanMessage(content=f"Find the shortest operation path to {target}")
        ],
        "problem": f"Reach {target} starting from 1.0",
        "steps": ["Init: val=1.0"],
        "current_estimate": 1.0,
        "target_value": target,
    }
    graph.invoke(initial_input, config=config)

    # 7. Configure and Run BeamSearchRunner
    beam_width = 2
    branch_factor = 3
    max_depth = 4

    print(
        f"\nConfiguration: Width (K)={beam_width}, "
        f"Branch Factor (B)={branch_factor}, Max Depth={max_depth}"
    )
    print(f"Goal: Starting from 1.0, converge to target {target}\n")

    runner = BeamSearchRunner(
        saver=saver,
        config=BeamSearchConfig(
            beam_width=beam_width,
            branch_factor=branch_factor,
            max_depth=max_depth,
            prune_discarded=True,
        ),
    )
    evaluator = HeuristicEvaluator(scorer=score_proximity)

    search_result = runner.run(
        graph=graph,
        initial_config=config,
        expand_fn=expand_reasoning_candidates,
        evaluator=evaluator,
    )

    # 8. Report Per-Depth Progression
    print("--- 1. Per-Depth Beam Search Progression ---")
    for summary in search_result.depth_history:
        print(
            f"  Depth {summary.depth}: Expanded={summary.candidates_expanded} | "
            f"Survivors={len(summary.survivors)} | "
            f"Pruned={len(summary.pruned)} | "
            f"Best Score={summary.best_score:.4f}"
        )

    # 9. Report Winning Trajectory
    winner = search_result.winner
    print("\n--- 2. Optimal Reasoning Trajectory ---")
    print(f"Depth Reached:            {search_result.depth_reached}")
    print(
        f"Total Explored:           {search_result.total_candidates_explored} branches"
    )
    print(f"Final Best Score:         {search_result.winner_score:.4f}")
    if winner.output:
        print(f"Final Estimated Value:    {winner.output.get('current_estimate')}")
        print("Step-by-Step Path:")
        for step in winner.output.get("steps", []):
            print(f"  → {step}")

    # 10. Visualize Pruned Janus DAG
    print("\n--- 3. Cleaned Janus Multiverse DAG (Mermaid) ---")
    print(saver.visualize(thread_id))

    print("\n✅ Multi-depth beam search reasoning completed successfully!")


if __name__ == "__main__":
    main()
