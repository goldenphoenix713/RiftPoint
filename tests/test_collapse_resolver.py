"""Comprehensive tests for evaluators, MultiverseResolver, and collapse workflows."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import (
    AsyncRiftCheckpointSaver,
    BranchResult,
    BranchSpec,
    ConsensusEvaluator,
    EvaluationResult,
    HeuristicEvaluator,
    JSONSchemaEvaluator,
    LLMJudgeEvaluator,
    MultiverseResolver,
    RiftCheckpointSaver,
    RiftRunner,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class MultiAgentState(TypedDict):
    """Test state for multiversal agent collapse."""

    messages: Annotated[list[BaseMessage], add_messages]
    strategy_used: str
    result_data: str
    confidence: float


def test_evaluation_result_properties() -> None:
    """Test EvaluationResult properties and defaults."""
    res_pass = EvaluationResult(score=0.9, reasoning="Excellent")
    assert res_pass.is_passing
    assert res_pass.score == 0.9
    assert res_pass.reasoning == "Excellent"
    assert res_pass.metadata == {}

    res_fail = EvaluationResult(score=0.0)
    assert not res_fail.is_passing


def test_heuristic_evaluator() -> None:
    """Test HeuristicEvaluator with various scoring functions."""
    evaluator = HeuristicEvaluator(
        lambda res: (res.output or {}).get("confidence", 0.0),
        name="confidence_scorer",
    )

    good_branch = BranchResult(
        branch_name="b_good",
        output={"confidence": 0.95, "text": "high quality"},
    )
    score_res = evaluator.evaluate(good_branch)
    assert score_res.score == 0.95
    assert score_res.is_passing

    # Test failing branch
    failed_branch = BranchResult(
        branch_name="b_fail",
        error=RuntimeError("crashed"),
    )
    score_fail = evaluator.evaluate(failed_branch)
    assert score_fail.score == 0.0
    assert not score_fail.is_passing

    # Test exception in scorer function
    def faulty_scorer(_res: BranchResult) -> float:
        msg = "Bug in scorer"
        raise ValueError(msg)

    faulty_eval = HeuristicEvaluator(faulty_scorer)
    faulty_res = faulty_eval.evaluate(good_branch)
    assert faulty_res.score == 0.0
    assert "Evaluation failed" in (faulty_res.reasoning or "")


def test_json_schema_evaluator() -> None:
    """Test JSONSchemaEvaluator with full, partial, and missing keys."""
    evaluator = JSONSchemaEvaluator(required_keys=["answer", "sources"])

    # Complete match
    res_full = BranchResult(
        branch_name="b1",
        output={"answer": "42", "sources": ["doc1"]},
    )
    score_full = evaluator.evaluate(res_full)
    assert score_full.score == 1.0
    assert "All required keys present" in (score_full.reasoning or "")

    # Partial match
    res_partial = BranchResult(
        branch_name="b2",
        output={"answer": "42"},
    )
    score_partial = evaluator.evaluate(res_partial)
    assert score_partial.score == 0.5
    assert "sources" in score_partial.metadata["missing_keys"]

    # Scoped channel
    scoped_eval = JSONSchemaEvaluator(
        required_keys=["query"], output_channel="metadata"
    )
    res_scoped = BranchResult(
        branch_name="b3",
        output={"metadata": {"query": "SELECT *"}, "result": "ok"},
    )
    assert scoped_eval.evaluate(res_scoped).score == 1.0


def test_consensus_evaluator() -> None:
    """Test ConsensusEvaluator for agreement across branches."""
    evaluator = ConsensusEvaluator(target_channel="strategy_used")

    branch_1 = BranchResult(branch_name="b1", output={"strategy_used": "sql"})
    branch_2 = BranchResult(branch_name="b2", output={"strategy_used": "sql"})
    branch_3 = BranchResult(branch_name="b3", output={"strategy_used": "web"})

    results_map = {"b1": branch_1, "b2": branch_2, "b3": branch_3}
    evaluator.fit(list(results_map.values()))

    # Majority (2/3 = ~0.667)
    score_1 = evaluator.evaluate(branch_1)
    score_3 = evaluator.evaluate(branch_3)

    assert score_1.score > 0.6
    assert score_3.score == 0.0


def test_llm_judge_evaluator() -> None:
    """Test LLMJudgeEvaluator with mock LLM judge callable."""

    def mock_judge(prompt: str) -> str:
        if "strategy_optimal" in prompt:
            return "SCORE: 0.95\nREASONING: Superior reasoning trajectory."
        return "SCORE: 0.40\nREASONING: Suboptimal approach."

    evaluator = LLMJudgeEvaluator(mock_judge)

    res_good = BranchResult(
        branch_name="good",
        output={"text": "strategy_optimal output"},
    )
    res_bad = BranchResult(
        branch_name="bad",
        output={"text": "other output"},
    )

    eval_good = evaluator.evaluate(res_good)
    eval_bad = evaluator.evaluate(res_bad)

    assert eval_good.score == 0.95
    assert "Superior reasoning" in (eval_good.reasoning or "")
    assert eval_bad.score == 0.40


def test_multiverse_resolver_collapse_sync() -> None:
    """Test synchronous collapse, state promotion, and branch pruning."""
    saver = RiftCheckpointSaver()
    runner = RiftRunner(saver)
    resolver = MultiverseResolver(saver)

    def agent_node(state: MultiAgentState) -> dict[str, Any]:
        strategy = state.get("strategy_used", "default")
        conf = 0.95 if strategy == "strategy_winner" else 0.40
        return {
            "messages": [AIMessage(content=f"Executed {strategy}")],
            "result_data": f"Data_{strategy}",
            "confidence": conf,
        }

    builder = StateGraph(MultiAgentState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    graph = builder.compile(checkpointer=saver)

    thread_id = "collapse-session-sync"
    initial_config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }

    # Initial graph run to seed main
    graph.invoke(
        {
            "messages": [HumanMessage(content="Start query")],
            "strategy_used": "init",
            "result_data": "start",
            "confidence": 0.0,
        },
        config=initial_config,
    )

    # Run speculative branches
    branch_specs = [
        BranchSpec(
            name="branch_a",
            input_data={"strategy_used": "strategy_loser"},
        ),
        BranchSpec(
            name="branch_winner",
            input_data={"strategy_used": "strategy_winner"},
        ),
    ]

    branch_results = runner.run_parallel_branches(graph, initial_config, branch_specs)

    evaluator = HeuristicEvaluator(
        lambda res: (res.output or {}).get("confidence", 0.0)
    )

    collapse_result = resolver.collapse(
        thread_id=thread_id,
        results=branch_results,
        evaluator=evaluator,
        prune_discarded=True,
    )

    assert collapse_result.winning_branch == "branch_winner"
    assert collapse_result.scores["branch_winner"].score == 0.95
    assert "branch_a" in collapse_result.pruned_branches
    assert "branch_winner" in collapse_result.pruned_branches

    # Verify canonical timeline now has the promoted winning state
    canonical_tuple = saver.get_tuple(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    assert canonical_tuple is not None
    assert (
        canonical_tuple.checkpoint["channel_values"]["result_data"]
        == "Data_strategy_winner"
    )


def test_multiverse_resolver_collapse_async() -> None:
    """Test asynchronous collapse workflow with AsyncRiftCheckpointSaver."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        runner = RiftRunner(saver)
        resolver = MultiverseResolver(saver)

        async def agent_node(state: MultiAgentState) -> dict[str, Any]:
            strategy = state.get("strategy_used", "default")
            await asyncio.sleep(0.001)
            conf = 0.99 if strategy == "async_best" else 0.50
            return {
                "messages": [AIMessage(content=f"Async {strategy}")],
                "result_data": f"Async_{strategy}",
                "confidence": conf,
            }

        builder = StateGraph(MultiAgentState)
        builder.add_node("agent", agent_node)
        builder.add_edge(START, "agent")
        graph = builder.compile(checkpointer=saver)

        thread_id = "collapse-session-async"
        initial_config: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
        }

        await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Start async")],
                "strategy_used": "init",
                "result_data": "none",
                "confidence": 0.0,
            },
            config=initial_config,
        )

        branch_specs = [
            BranchSpec(
                name="async_suboptimal",
                input_data={"strategy_used": "async_sub"},
            ),
            BranchSpec(
                name="async_optimal",
                input_data={"strategy_used": "async_best"},
            ),
        ]

        branch_results = await runner.arun_parallel_branches(
            graph, initial_config, branch_specs
        )

        evaluator = HeuristicEvaluator(
            lambda res: (res.output or {}).get("confidence", 0.0)
        )

        collapse_result = await resolver.acollapse(
            thread_id=thread_id,
            results=branch_results,
            evaluator=evaluator,
            prune_discarded=True,
        )

        assert collapse_result.winning_branch == "async_optimal"
        assert collapse_result.scores["async_optimal"].score == 0.99

        # Canonical timeline verification
        canonical_tuple = await saver.aget_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        assert canonical_tuple is not None
        assert (
            canonical_tuple.checkpoint["channel_values"]["result_data"]
            == "Async_async_best"
        )

    asyncio.run(_run())


def test_collapse_tie_breaker() -> None:
    """Test custom tie-breaking logic when scores are tied."""
    saver = RiftCheckpointSaver()
    resolver = MultiverseResolver(saver)

    res_1 = BranchResult(
        branch_name="branch_beta",
        output={"score": 0.8},
    )
    res_2 = BranchResult(
        branch_name="branch_alpha",
        output={"score": 0.8},
    )

    scores = {
        "branch_beta": EvaluationResult(score=0.8),
        "branch_alpha": EvaluationResult(score=0.8),
    }

    # Custom tie breaker choosing longest name
    winner = resolver.select_winner(
        scores=scores,
        results={"branch_beta": res_1, "branch_alpha": res_2},
        tie_breaker=lambda candidates: max(candidates, key=len),
    )
    assert winner == "branch_alpha"
