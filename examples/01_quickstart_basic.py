"""Quickstart example demonstrating basic RiftPoint multiversal state management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypedDict

from langgraph.graph import START, StateGraph

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

from riftpoint import (
    BranchSpec,
    HeuristicEvaluator,
    MultiverseResolver,
    RiftCheckpointSaver,
    RiftRunner,
)


# 1. Define Agent State Schema
class AgentState(TypedDict):
    """Simple agent state."""

    step_count: int
    messages: Annotated[list[str], lambda left, right: left + right]


# 2. Define Graph Nodes
def initial_reasoning_node(state: AgentState) -> dict[str, object]:
    """Base reasoning node on canonical timeline."""
    return {
        "step_count": state["step_count"] + 1,
        "messages": ["Canonical: Initial goal formulated."],
    }


def step_node(state: AgentState) -> dict[str, object]:
    """Step execution node."""
    return {
        "step_count": state["step_count"] + 1,
        "messages": [f"Step {state['step_count'] + 1} completed."],
    }


def main() -> None:
    """Run the quickstart demonstration."""
    print("=" * 70)
    print("🌌 RiftPoint: High-Performance Multiversal State Management")
    print("=" * 70)

    # 3. Build & Compile StateGraph with RiftCheckpointSaver
    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("reasoning", initial_reasoning_node)
    graph_builder.add_node("step", step_node)
    graph_builder.add_edge(START, "reasoning")
    graph_builder.add_edge("reasoning", "step")

    saver = RiftCheckpointSaver()
    graph = graph_builder.compile(checkpointer=saver)

    thread_id = "session-demo-01"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 4. Run Canonical Baseline Execution
    print("\n--- 1. Canonical Timeline Execution ---")
    initial_input: AgentState = {"step_count": 0, "messages": []}
    state = graph.invoke(initial_input, config=config)
    print(f"Canonical Step Count: {state['step_count']}")
    print(f"Canonical Messages:   {state['messages']}")

    # 5. Multiversal Branching with RiftRunner
    print("\n--- 2. Speculative Parallel Branching ---")
    runner = RiftRunner(saver=saver)

    # Launch two alternative speculative trajectories from current checkpoint
    results = runner.run_parallel_branches(
        graph=graph,
        initial_config=config,
        branch_specs=[
            BranchSpec(
                name="fast_path",
                input_data={"messages": ["Candidate Fast: direct heuristic."]},
            ),
            BranchSpec(
                name="deep_path",
                input_data={"messages": ["Candidate Deep: detailed analysis."]},
            ),
        ],
    )

    for branch_name, res in results.items():
        print(f"Branch '{branch_name}': Success={res.is_success}, State={res.output}")

    # 6. Generate Mermaid Diagram of the Multiverse DAG
    print("\n--- 3. Janus Multiverse DAG (Mermaid) ---")
    mermaid_chart = saver.visualize(thread_id)
    print(mermaid_chart)

    # 7. Evaluate and Collapse Candidate Branches
    print("\n--- 4. Multiverse Collapse ---")
    evaluator = HeuristicEvaluator(
        scorer=lambda res: 1.0 if "deep_path" in str(res.output) else 0.5
    )
    resolver = MultiverseResolver(saver=saver)

    collapse_result = resolver.collapse(
        thread_id=thread_id,
        results=results,
        evaluator=evaluator,
        prune_discarded=True,
    )

    winner = collapse_result.winning_branch
    print(f"Winning Branch:      {winner}")
    print(f"Winning Score:       {collapse_result.scores[winner].score}")
    print(f"Pruned Branches:     {collapse_result.pruned_branches}")

    # 8. Verify Canonical Timeline Promotion
    final_tuple = saver.get_tuple(config)
    if final_tuple:
        print("\n--- 5. Promoted Canonical State ---")
        print(f"Channel Values: {final_tuple.checkpoint['channel_values']}")

    # 9. Session Export / Import
    print("\n--- 6. Session State Serialization ---")
    json_export = saver.export_session(thread_id)
    print(f"Exported JSON ({len(json_export)} chars): {json_export[:120]}...")

    # Restore in fresh saver
    new_saver = RiftCheckpointSaver()
    new_saver.import_session(json_export)
    restored_tuple = new_saver.get_tuple(config)
    is_restored = bool(
        restored_tuple is not None
        and final_tuple is not None
        and restored_tuple.config == final_tuple.config
    )
    print(f"Restored into new checkpointer successfully: {is_restored}")
    print("\n🎉 RiftPoint demonstration completed successfully.")


if __name__ == "__main__":
    main()
