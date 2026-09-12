"""Example demonstrating speculative tool execution with JSON schema validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

from riftpoint import (
    BranchSpec,
    JSONSchemaEvaluator,
    MultiverseResolver,
    RiftCheckpointSaver,
    RiftRunner,
)


# 1. State Schema
class ResearchAgentState(TypedDict):
    """Research agent state schema."""

    query: str
    strategy: Literal["none", "web_search", "vector_db", "malformed"]
    tool_output: dict[str, Any] | str
    confidence: float
    history: Annotated[list[str], lambda left, right: left + right]


# 2. Graph Nodes
def ingest_query_node(state: ResearchAgentState) -> dict[str, Any]:
    """Capture user query on canonical timeline."""
    return {
        "history": [f"Canonical Query: {state['query']}"],
    }


def web_search_node(state: ResearchAgentState) -> dict[str, Any]:
    """Execute high-precision web search API."""
    _ = state
    payload = {
        "source": "web_search",
        "answer": "Janus-Tachyon-RS enables O(1) multiversal state forking.",
        "confidence": 0.98,
    }
    return {
        "tool_output": payload,
        "confidence": 0.98,
        "history": ["Executed Web Search API tool."],
    }


def vector_db_node(state: ResearchAgentState) -> dict[str, Any]:
    """Execute local vector retrieval tool."""
    _ = state
    payload = {
        "source": "vector_db",
        "data": "Found 3 matching vectors in local index.",
    }
    return {
        "tool_output": payload,
        "confidence": 0.70,
        "history": ["Executed Vector Store Retrieval tool."],
    }


def malformed_tool_node(state: ResearchAgentState) -> dict[str, Any]:
    """Simulate tool returning malformed plain text."""
    _ = state
    return {
        "tool_output": "Error: Unstructured raw dump without JSON format",
        "confidence": 0.10,
        "history": ["Executed Legacy Tool."],
    }


def route_strategy(
    state: ResearchAgentState,
) -> str:
    """Conditional router selecting tool execution branch."""
    strategy = state.get("strategy", "none")
    if strategy in ("web_search", "vector_db", "malformed"):
        return strategy
    return END


def main() -> None:
    """Run speculative tool calling and validation example."""
    print("=" * 75)
    print("🔬 RiftPoint: Speculative Tool Calling & Schema Validation")
    print("=" * 75)

    # 3. Build & Compile StateGraph
    builder = StateGraph(ResearchAgentState)
    builder.add_node("ingest", ingest_query_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("vector_db", vector_db_node)
    builder.add_node("malformed", malformed_tool_node)

    builder.add_edge(START, "ingest")
    builder.add_conditional_edges(
        "ingest",
        route_strategy,
        {
            "web_search": "web_search",
            "vector_db": "vector_db",
            "malformed": "malformed",
            END: END,
        },
    )
    builder.add_edge("web_search", END)
    builder.add_edge("vector_db", END)
    builder.add_edge("malformed", END)

    saver = RiftCheckpointSaver()
    graph = builder.compile(checkpointer=saver)

    thread_id = "research-session-42"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # 4. Canonical Baseline Step
    print("\n--- 1. Ingest Query on Canonical Reality ---")
    initial_input: ResearchAgentState = {
        "query": "How does RiftPoint achieve near-zero copy state branching?",
        "strategy": "none",
        "tool_output": "",
        "confidence": 0.0,
        "history": [],
    }
    initial_state = graph.invoke(initial_input, config=config)
    print(f"Canonical History: {initial_state['history']}")

    # 5. Speculative Parallel Branching with Divergent Tool Strategies
    print("\n--- 2. Speculative Tool Execution across Parallel Branches ---")
    runner = RiftRunner(saver=saver)

    specs = [
        BranchSpec(
            name="search_engine_branch",
            input_data={"strategy": "web_search"},
        ),
        BranchSpec(
            name="vector_db_branch",
            input_data={"strategy": "vector_db"},
        ),
        BranchSpec(
            name="malformed_branch",
            input_data={"strategy": "malformed"},
        ),
    ]

    results = runner.run_parallel_branches(
        graph=graph,
        initial_config=config,
        branch_specs=specs,
    )

    for name, res in results.items():
        hist = res.output.get("history") if res.output else None
        tool_res = res.output.get("tool_output") if res.output else None
        print(f"\nBranch '{name}':")
        print(f"  Success: {res.is_success}")
        print(f"  History: {hist}")
        print(f"  Output:  {tool_res}")

    # 6. JSON Schema Validation & Collapse
    print("\n--- 3. Evaluating Candidate Timelines with JSONSchemaEvaluator ---")
    schema_evaluator = JSONSchemaEvaluator(
        required_keys=["source", "answer", "confidence"],
        output_channel="tool_output",
    )

    resolver = MultiverseResolver(saver=saver)

    collapse_result = resolver.collapse(
        thread_id=thread_id,
        results=results,
        evaluator=schema_evaluator,
        prune_discarded=True,
    )

    winner = collapse_result.winning_branch
    print(f"\n🏆 Winning Trajectory: {winner}")
    print(f"   Score:              {collapse_result.scores[winner].score:.2f}")
    print(f"   Pruned Branches:    {collapse_result.pruned_branches}")

    for b_name, eval_res in collapse_result.scores.items():
        print(
            f"   - Branch '{b_name}' score: {eval_res.score:.2f} ({eval_res.metadata})"
        )

    # 7. Final Canonical Verification
    canonical_tuple = saver.get_tuple(config)
    if canonical_tuple:
        print("\n--- 4. Final Canonical Reality State ---")
        channel_vals = canonical_tuple.checkpoint["channel_values"]
        print(f"Promoted History:     {channel_vals.get('history')}")
        print(f"Promoted Tool Output: {channel_vals.get('tool_output')}")
        print(f"Confidence:           {channel_vals.get('confidence')}")

    print("\n✅ Speculative tool evaluation and multiverse collapse succeeded.")


if __name__ == "__main__":
    main()
