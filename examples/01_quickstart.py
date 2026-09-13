"""01_quickstart.py — Minimal LangGraph integration with RiftPoint.

Demonstrates:
1. Initializing a LangGraph StateGraph with RiftCheckpointSaver.
2. Running canonical execution.
3. Spawning zero-copy speculative branches via RiftRunner.
4. Visualizing the execution DAG as Mermaid.
5. Collapsing to the winning timeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypedDict

from langgraph.graph import START, StateGraph

from riftpoint import (
    BranchSpec,
    HeuristicEvaluator,
    MultiverseResolver,
    RiftCheckpointSaver,
    RiftRunner,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from riftpoint.runner.multiverse import BranchResult


# 1. Define Agent State Schema
class AgentState(TypedDict):
    """Simple agent state."""

    step_count: int
    messages: Annotated[list[str], lambda left, right: left + right]


# 2. Define Graph Nodes
def reasoning_node(state: AgentState) -> dict[str, object]:
    """Base reasoning node on canonical timeline."""
    return {
        "step_count": state["step_count"] + 1,
        "messages": ["Canonical: Initial reasoning complete."],
    }


def action_node(state: AgentState) -> dict[str, object]:
    """Action execution node."""
    return {
        "step_count": state["step_count"] + 1,
        "messages": [f"Step {state['step_count'] + 1}: Action performed."],
    }


def main() -> None:
    """Run the quickstart demonstration."""
    print("=" * 70)
    print("🌌 RiftPoint: Minimal LangGraph Quickstart")
    print("=" * 70)

    # 3. Build & Compile StateGraph with RiftCheckpointSaver
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("reasoning", reasoning_node)
    graph_builder.add_node("action", action_node)
    graph_builder.add_edge(START, "reasoning")
    graph_builder.add_edge("reasoning", "action")

    saver = RiftCheckpointSaver()
    graph = graph_builder.compile(checkpointer=saver)

    thread_id = "quickstart-session"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 4. Canonical Timeline Execution
    print("\n--- 1. Canonical Execution ---")
    initial_input: AgentState = {"step_count": 0, "messages": []}
    state = graph.invoke(initial_input, config=config)
    print(f"Canonical Step Count: {state['step_count']}")
    print(f"Canonical Messages:   {state['messages']}")

    # 5. Multiversal Branching with RiftRunner
    print("\n--- 2. Speculative Parallel Branching ---")
    runner = RiftRunner(saver=saver)

    results = runner.run_parallel_branches(
        graph=graph,
        initial_config=config,
        branch_specs=[
            BranchSpec(
                name="fast_path",
                input_data={"messages": ["Fast heuristic candidate."]},
            ),
            BranchSpec(
                name="deep_path",
                input_data={
                    "messages": ["Deep analytical candidate with verification."]
                },
            ),
        ],
    )

    for branch_name, res in results.items():
        msgs = res.output.get("messages") if res.output else None
        print(f"  Branch '{branch_name}': Success={res.is_success}, Messages={msgs}")

    # 6. Generate Mermaid Diagram of the Multiverse DAG
    print("\n--- 3. Janus Multiverse DAG (Mermaid) ---")
    mermaid_chart = saver.visualize(thread_id)
    print(mermaid_chart)

    # 7. Evaluate and Collapse Candidate Branches
    print("\n--- 4. Multiverse Collapse ---")

    def score_by_detail(result: BranchResult) -> float:
        if not result.is_success or result.output is None:
            return 0.0
        msgs = result.output.get("messages", [])
        return float(len(" ".join(msgs)))

    resolver = MultiverseResolver(saver=saver)
    collapse_summary = resolver.collapse(
        thread_id=thread_id,
        results=results,
        evaluator=HeuristicEvaluator(scorer=score_by_detail),
        prune_discarded=True,
    )

    winner = collapse_summary.winning_branch
    winner_score = collapse_summary.scores[winner].score
    print(f"Winning Branch:       {winner}")
    print(f"Winner Score:         {winner_score:.2f}")
    print(f"Pruned Discarded:     {collapse_summary.pruned_branches}")

    # 8. Resume Canonical Graph on Winner Trajectory
    print("\n--- 5. Resumed Canonical Graph ---")
    winning_state = saver.get_tuple(config)
    if winning_state:
        print(f"Active Canonical State: {winning_state.checkpoint['channel_values']}")
    print("\n✅ Quickstart completed successfully!")


if __name__ == "__main__":
    main()
