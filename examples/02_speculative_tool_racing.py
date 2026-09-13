"""02_speculative_tool_racing.py — Speculative Tool Execution with @speculative_node.

Demonstrates:
1. Using the `@speculative_node` decorator on a LangGraph node.
2. Racing multiple tool strategies concurrently (Cache vs RAG vs Web Search).
3. Automatically evaluating candidates and committing the highest-scoring output.
4. Inspecting speculative execution metadata (`__speculative_meta__`).
5. Demonstrating graceful fallback when candidate tools fail.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import (
    HeuristicEvaluator,
    RiftCheckpointSaver,
    SpeculativeRaceMeta,
    speculative_node,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from riftpoint.runner.multiverse import BranchResult


# 1. State Schema
class ResearchAgentState(TypedDict, total=False):
    """Agent state for speculative research."""

    messages: Annotated[list[Any], add_messages]
    query: str
    answer: str
    source: str
    confidence: float
    __speculative_meta__: SpeculativeRaceMeta | None


# 2. Candidate Tool Functions
def fetch_cache(state: dict[str, Any]) -> dict[str, Any]:
    """Simulated fast in-memory cache lookup."""
    query = state.get("query", "")
    time.sleep(0.01)  # Simulate 10ms latency
    if "RiftPoint" in query:
        return {
            "answer": "RiftPoint is a high-performance multiversal checkpointer.",
            "source": "cache",
            "confidence": 0.85,
        }
    return {
        "answer": "Cache miss",
        "source": "cache",
        "confidence": 0.0,
    }


def fetch_vector_rag(state: dict[str, Any]) -> dict[str, Any]:
    """Simulated Vector RAG retrieval."""
    query = state.get("query", "")
    time.sleep(0.03)  # Simulate 30ms latency
    return {
        "answer": f"Vector RAG retrieved detailed specs for query '{query}'.",
        "source": "vector_rag",
        "confidence": 0.92,
    }


def fetch_live_web(state: dict[str, Any]) -> dict[str, Any]:
    """Simulated live web search."""
    query = state.get("query", "")
    time.sleep(0.05)  # Simulate 50ms latency
    return {
        "answer": f"Live web search found latest release info for '{query}'.",
        "source": "web_search",
        "confidence": 0.96,
    }


# 3. Custom Heuristic Evaluator Scorer
def score_tool_candidate(result: BranchResult) -> float:
    """Score candidate based on confidence and output validity."""
    if not result.is_success or not result.output:
        return 0.0
    return float(result.output.get("confidence", 0.0))


def main() -> None:
    """Run speculative tool racing demonstration."""
    print("=" * 75)
    print("🚀 RiftPoint: Speculative Tool Racing with @speculative_node")
    print("=" * 75)

    evaluator = HeuristicEvaluator(scorer=score_tool_candidate)

    # 4. Define Speculative Node with Decorator
    @speculative_node(
        tools=[fetch_cache, fetch_vector_rag, fetch_live_web],
        evaluator=evaluator,
        max_workers=3,
        fallback_on_all_fail=True,
    )
    def research_node(state: ResearchAgentState) -> dict[str, Any]:
        """Fallback node if all speculative candidate tools fail."""
        return {
            "answer": f"Fallback default response for: {state.get('query')}",
            "source": "fallback",
            "confidence": 0.1,
        }

    def summarize_node(state: ResearchAgentState) -> dict[str, Any]:
        """Synthesize final response."""
        content = (
            f"[{state['source'].upper()}] {state['answer']} "
            f"(confidence: {state['confidence']:.2f})"
        )
        return {
            "messages": [AIMessage(content=content)],
            "__speculative_meta__": state.get("__speculative_meta__"),
        }

    # 5. Build and Compile Graph
    builder = StateGraph(ResearchAgentState)
    builder.add_node("research", research_node)
    builder.add_node("summarize", summarize_node)
    builder.add_edge(START, "research")
    builder.add_edge("research", "summarize")

    saver = RiftCheckpointSaver()
    graph = builder.compile(checkpointer=saver)

    # 6. Execute Speculative Race
    print("\n--- 1. Executing Concurrent Tool Race ---")
    config: RunnableConfig = {
        "configurable": {"thread_id": "spec-race-01", "checkpoint_ns": ""}
    }
    input_state: ResearchAgentState = {
        "messages": [HumanMessage(content="What is the latest status of RiftPoint?")],
        "query": "RiftPoint status",
        "answer": "",
        "source": "",
        "confidence": 0.0,
    }

    start_time = time.perf_counter()
    result = graph.invoke(input_state, config=config)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"Elapsed Time:         {elapsed_ms:.1f} ms")
    print(f"Winning Tool Source:  {result['source']}")
    print(f"Winning Confidence:   {result['confidence']:.2f}")
    print(f"Synthesized Output:   {result['messages'][-1].content}")

    # 7. Inspect Speculative Metadata
    meta: SpeculativeRaceMeta | None = result.get("__speculative_meta__")
    if meta:
        print("\n--- 2. Speculative Race Metadata ---")
        print(f"Winner Tool:          {meta.winner_tool}")
        print(f"All Tools Succeeded:  {meta.all_succeeded}")
        print(f"Fallback Used:        {meta.used_fallback}")
        print("Candidate Scores:")
        for tool_name, score in sorted(
            meta.scores.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  • {tool_name:20s}: score = {score:.2f}")

    print("\n✅ Speculative tool racing completed successfully!")


if __name__ == "__main__":
    main()
