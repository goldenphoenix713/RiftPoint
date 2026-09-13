"""Comprehensive tests for the @speculative_node decorator."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import (
    HeuristicEvaluator,
    RiftCheckpointSaver,
    speculative_node,
)
from riftpoint.decorators import SpeculativeRaceMeta

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from riftpoint.runner.multiverse import BranchResult


def _score_by_confidence(result: BranchResult) -> float:
    """Score based on confidence field in output."""
    if not result.is_success or result.output is None:
        return 0.0
    return float(result.output.get("confidence", 0.0))


# --- Tool functions for testing ---


def tool_web_search(state: dict[str, Any]) -> dict[str, Any]:
    """Simulated web search tool returning high confidence."""
    return {
        "answer": "Web result for: " + state.get("query", ""),
        "source": "web",
        "confidence": 0.95,
    }


def tool_database(state: dict[str, Any]) -> dict[str, Any]:
    """Simulated database tool returning medium confidence."""
    return {
        "answer": "DB result for: " + state.get("query", ""),
        "source": "database",
        "confidence": 0.75,
    }


def tool_cache(state: dict[str, Any]) -> dict[str, Any]:
    """Simulated cache tool returning low confidence."""
    return {
        "answer": "Cached result",
        "source": "cache",
        "confidence": 0.50,
    }


def tool_broken(state: dict[str, Any]) -> dict[str, Any]:
    """Tool that always crashes."""
    msg = "Tool connection timeout"
    raise RuntimeError(msg)


async def async_tool_fast(state: dict[str, Any]) -> dict[str, Any]:
    """Async tool with high confidence."""
    await asyncio.sleep(0.001)
    return {
        "answer": "Fast async result",
        "source": "async_fast",
        "confidence": 0.9,
    }


async def async_tool_slow(state: dict[str, Any]) -> dict[str, Any]:
    """Async tool with lower confidence."""
    await asyncio.sleep(0.001)
    return {
        "answer": "Slow async result",
        "source": "async_slow",
        "confidence": 0.6,
    }


# --- Tests ---


def test_speculative_node_sync_basic() -> None:
    """Three tools raced synchronously, highest confidence wins."""
    evaluator = HeuristicEvaluator(scorer=_score_by_confidence)

    @speculative_node(
        tools=[tool_web_search, tool_database, tool_cache],
        evaluator=evaluator,
    )
    def research(state: dict[str, Any]) -> dict[str, Any]:
        """Fallback research function."""
        return {"answer": "fallback", "source": "fallback", "confidence": 0.0}

    state = {"query": "What is RiftPoint?"}
    result = research(state)

    assert result["source"] == "web"
    assert result["confidence"] == 0.95
    assert "__speculative_meta__" in result

    meta = result["__speculative_meta__"]
    assert isinstance(meta, SpeculativeRaceMeta)
    assert meta.winner_tool == "tool_web_search"
    assert len(meta.scores) == 3
    assert meta.all_succeeded
    assert not meta.used_fallback


def test_speculative_node_async_basic() -> None:
    """Async tools raced with asyncio.gather, best scorer wins."""
    evaluator = HeuristicEvaluator(scorer=_score_by_confidence)

    @speculative_node(
        tools=[async_tool_fast, async_tool_slow],
        evaluator=evaluator,
    )
    async def async_research(state: dict[str, Any]) -> dict[str, Any]:
        """Async fallback."""
        return {"answer": "async_fallback", "source": "fallback", "confidence": 0.0}

    async def _run() -> None:
        state = {"query": "async test"}
        result = await async_research(state)

        assert result["source"] == "async_fast"
        assert result["confidence"] == 0.9
        meta = result["__speculative_meta__"]
        assert isinstance(meta, SpeculativeRaceMeta)
        assert meta.winner_tool == "async_tool_fast"
        assert not meta.used_fallback

    asyncio.run(_run())


def test_speculative_node_error_isolation() -> None:
    """One tool crashes, others succeed — crash doesn't break the race."""
    evaluator = HeuristicEvaluator(scorer=_score_by_confidence)

    @speculative_node(
        tools=[tool_database, tool_broken, tool_cache],
        evaluator=evaluator,
    )
    def research(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "fallback", "source": "fallback", "confidence": 0.0}

    state = {"query": "error test"}
    result = research(state)

    # Database has highest confidence among successful tools
    assert result["source"] == "database"
    assert result["confidence"] == 0.75
    meta = result["__speculative_meta__"]
    assert isinstance(meta, SpeculativeRaceMeta)
    assert not meta.all_succeeded
    assert meta.scores["tool_broken"] == 0.0


def test_speculative_node_all_fail_fallback() -> None:
    """All tools fail, original function used as fallback."""
    evaluator = HeuristicEvaluator(scorer=_score_by_confidence)

    @speculative_node(
        tools=[tool_broken, tool_broken],
        evaluator=evaluator,
        fallback_on_all_fail=True,
    )
    def research(state: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "safe fallback", "source": "fallback", "confidence": 0.1}

    state = {"query": "all fail test"}
    result = research(state)

    assert result["source"] == "fallback"
    assert result["answer"] == "safe fallback"
    meta = result["__speculative_meta__"]
    assert isinstance(meta, SpeculativeRaceMeta)
    assert meta.used_fallback
    assert not meta.all_succeeded


def test_speculative_node_preserves_metadata() -> None:
    """Verify functools.wraps preserves name and docstring."""
    evaluator = HeuristicEvaluator(scorer=_score_by_confidence)

    @speculative_node(
        tools=[tool_cache],
        evaluator=evaluator,
    )
    def my_research_node(state: dict[str, Any]) -> dict[str, Any]:
        """This is my research node docstring."""
        return {"answer": "fallback"}

    assert my_research_node.__name__ == "my_research_node"
    assert my_research_node.__doc__ == "This is my research node docstring."


class SpecNodeState(TypedDict):
    """State for LangGraph integration test."""

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    answer: str
    source: str
    confidence: float


def test_speculative_node_in_langgraph() -> None:
    """Full integration: decorated node in a compiled StateGraph."""
    saver = RiftCheckpointSaver()
    evaluator = HeuristicEvaluator(scorer=_score_by_confidence)

    @speculative_node(
        tools=[tool_web_search, tool_database],
        evaluator=evaluator,
    )
    def speculative_research(state: SpecNodeState) -> dict[str, Any]:
        return {"answer": "fallback", "source": "fallback", "confidence": 0.0}

    def respond_node(state: SpecNodeState) -> dict[str, Any]:
        return {
            "messages": [AIMessage(content=f"Answer: {state['answer']}")],
        }

    builder = StateGraph(SpecNodeState)
    builder.add_node("research", speculative_research)
    builder.add_node("respond", respond_node)
    builder.add_edge(START, "research")
    builder.add_edge("research", "respond")
    graph = builder.compile(checkpointer=saver)

    config: RunnableConfig = {
        "configurable": {"thread_id": "spec-node-test", "checkpoint_ns": ""}
    }
    result = graph.invoke(
        {
            "messages": [HumanMessage(content="Tell me about RiftPoint")],
            "query": "What is RiftPoint?",
            "answer": "",
            "source": "",
            "confidence": 0.0,
        },
        config=config,
    )

    # Web search should win (confidence=0.95)
    assert result["source"] == "web"
    assert result["confidence"] == 0.95
    assert "Web result" in result["answer"]
