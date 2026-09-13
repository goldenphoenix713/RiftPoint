"""Comprehensive edge-case and fault tolerance tests for Phase 3 Hardening.

Targets full branch and line coverage across:
- BaseRiftSaver & checkpointer internals (snapshots, serde, delete_thread, cross-thread)
- AsyncRiftCheckpointSaver & SyncRiftCheckpointSaver edge cases
- Evaluators (LLMJudge parsing, Consensus modal edge cases, JSONSchema validation)
- Speculative Node decorator (sync in async, all-fail without fallback, exception paths)
- BeamSearchRunner (expansion exceptions, zero candidate termination, depth bounds)
- Visualization & plotting with file export
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
from langgraph.checkpoint.base import (
    Checkpoint,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.graph import START, StateGraph

from riftpoint import (
    AsyncRiftCheckpointSaver,
    BeamSearchConfig,
    BeamSearchRunner,
    BranchResult,
    BranchSpec,
    ConsensusEvaluator,
    EvaluationResult,
    HeuristicEvaluator,
    JSONSchemaEvaluator,
    LLMJudgeEvaluator,
    MultiverseResolver,
    RiftCheckpointSaver,
    SpeculativeNodeConfig,
    export_session_bytes,
    export_session_json,
    import_session_bytes,
    import_session_json,
    plot_multiverse,
    speculative_node,
)
from riftpoint.checkpointer.base import BaseRiftSaver
from riftpoint.runner.branch import BranchManager
from riftpoint.runner.multiverse import RiftRunner

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.runnables import RunnableConfig


class SimpleState(TypedDict):
    """Test state."""

    val: int
    text: str


def _make_chk(
    chk_id: str, values: dict[str, Any] | None = None
) -> tuple[Checkpoint, CheckpointMetadata]:
    chk: Checkpoint = empty_checkpoint()
    chk["id"] = chk_id
    chk["channel_values"] = values or {}
    chk["channel_versions"] = dict.fromkeys(values or {}, 1)
    meta: CheckpointMetadata = {"step": 1, "source": "input"}
    return chk, meta


# ==============================================================================
# 1. Base Checkpointer Edge Cases & Snapshot SerDe
# ==============================================================================


def test_checkpointer_delete_thread_and_namespace() -> None:
    """Verify thread and namespace deletion purges storage and blobs."""
    saver = RiftCheckpointSaver()
    thread_id = "purge-thread"
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "custom_ns",
            "checkpoint_id": "cp-1",
        }
    }

    chk, meta = _make_chk("cp-1", {"val": 42})
    saver.put(config, chk, meta, {"val": 1})
    saver.put_writes(config, [("channel", "write_val")], "task-1")

    # Verify presence
    assert saver.get_session_storage(thread_id)
    assert saver.get_session_writes(thread_id)

    # Delete namespace
    saver.delete_namespace(thread_id, "custom_ns")
    assert not saver.get_session_storage(thread_id).get("custom_ns")

    # Delete thread
    saver.delete_thread(thread_id)
    assert not saver.get_session_storage(thread_id)
    assert not saver.get_session_writes(thread_id)


def test_async_checkpointer_adelete_thread() -> None:
    """Verify async delete thread and aget_tuple edge cases."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        thread_id = "async-purge-thread"
        config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "cp-async-1",
            }
        }

        chk, meta = _make_chk("cp-async-1", {"val": 100})
        await saver.aput(config, chk, meta, {"val": 1})
        await saver.aput_writes(config, [("channel", "async_val")], "task-async-1")

        # Test aget_tuple missing branch
        missing_cfg: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "non_existent_ns",
            }
        }
        assert await saver.aget_tuple(missing_cfg) is None

        missing_cp_cfg: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "missing-cp-id",
            }
        }
        assert await saver.aget_tuple(missing_cp_cfg) is None

        assert saver.get_session_storage(thread_id)
        await saver.adelete_thread(thread_id)
        assert not saver.get_session_storage(thread_id)

    asyncio.run(_run())


def test_snapshot_export_and_import_edge_cases(tmp_path: Path) -> None:
    """Verify snapshot saving, loading (JSON and bytes), and error handling."""
    saver = RiftCheckpointSaver()
    thread_id = "snap-thread"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    chk, meta = _make_chk("cp-snap-1", {"val": 999})
    saver.put(config, chk, meta, {"val": 1})

    # JSON export / import
    snap_json_file = str(tmp_path / "snapshot.json")
    export_session_json(saver, thread_id, snap_json_file)

    new_saver_json = RiftCheckpointSaver()
    import_session_json(new_saver_json, snap_json_file)
    restored_cp = new_saver_json.get_tuple(config)
    assert restored_cp is not None
    assert restored_cp.checkpoint["channel_values"]["val"] == 999

    # Bytes export / import
    snap_bytes = export_session_bytes(saver, thread_id)
    new_saver_bin = RiftCheckpointSaver()
    import_session_bytes(new_saver_bin, snap_bytes)
    restored_bin_cp = new_saver_bin.get_tuple(config)
    assert restored_bin_cp is not None
    assert restored_bin_cp.checkpoint["channel_values"]["val"] == 999

    # Non-existent file path
    with pytest.raises(FileNotFoundError):
        import_session_json(new_saver_json, tmp_path / "missing.json")

    # Invalid JSON string
    with pytest.raises(json.JSONDecodeError):
        import_session_json(new_saver_json, "invalid json {")


def test_base_saver_tag_moment_exception_resilience() -> None:
    """Verify base saver handles tagging exceptions gracefully."""
    saver = RiftCheckpointSaver()
    # Trigger _record_janus_checkpoint with invalid types / error handling
    saver._record_janus_checkpoint("err-thread", "cp-1", {"step": 1})
    # Copy checkpoint non-existent
    assert not saver.copy_checkpoint_entry_cross_thread(
        from_thread_id="missing-t1",
        from_namespace="",
        to_thread_id="missing-t2",
        to_namespace="",
        checkpoint_id="missing-cp",
    )


def test_commit_branch_to_canonical_empty_error() -> None:
    """Verify error raised when committing from a non-existent branch."""
    saver = RiftCheckpointSaver()
    thread_id = "commit-err-thread"

    with pytest.raises(ValueError, match="No checkpoint found in branch"):
        saver.commit_branch_to_canonical(thread_id, "non_existent_branch")


def test_commit_branch_to_canonical_namespaced_fallback() -> None:
    """Verify commit_branch_to_canonical falls back to thread_id:branch namespace."""
    saver = RiftCheckpointSaver()
    thread_id = "ns-commit-thread"
    branch_name = "test_branch"
    branch_ns = f"branch:{branch_name}"

    # Put checkpoint directly in thread_id / branch_ns
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": branch_ns}
    }
    chk, meta = _make_chk("cp-branch-ns-1", {"val": 777})
    saver.put(config, chk, meta, {"val": 1})

    res_config = saver.commit_branch_to_canonical(thread_id, branch_name)
    assert res_config["configurable"]["checkpoint_id"] == "cp-branch-ns-1"


# ==============================================================================
# 2. Evaluator Edge Cases
# ==============================================================================


def test_heuristic_evaluator_returning_evaluation_result() -> None:
    """Verify HeuristicEvaluator when scorer returns EvaluationResult object."""

    def custom_scorer(result: BranchResult) -> EvaluationResult:
        _ = result
        return EvaluationResult(score=0.88, reasoning="Custom object returned")

    evaluator = HeuristicEvaluator(scorer=custom_scorer)
    dummy_res = BranchResult(branch_name="b1", output={"val": 1})
    res = evaluator.evaluate(dummy_res)
    assert res.score == 0.88
    assert res.reasoning == "Custom object returned"


def test_json_schema_evaluator_edge_cases() -> None:
    """Test JSONSchemaEvaluator with empty keys, failed branch, and non-dict channel."""
    # Empty schema
    empty_eval = JSONSchemaEvaluator(required_keys=[])
    success_res = BranchResult(branch_name="b1", output={"a": 1})
    assert empty_eval.evaluate(success_res).score == 1.0

    # Failed branch
    fail_res = BranchResult(branch_name="b2", error=RuntimeError("Tool failed"))
    schema_eval = JSONSchemaEvaluator(required_keys=["key1"])
    assert schema_eval.evaluate(fail_res).score == 0.0

    # Non-dict channel
    channel_eval = JSONSchemaEvaluator(
        required_keys=["key1"], output_channel="sub_data"
    )
    non_dict_res = BranchResult(branch_name="b3", output={"sub_data": "not a dict"})
    res = channel_eval.evaluate(non_dict_res)
    assert res.score == 0.0
    assert "not a dictionary" in str(res.reasoning)


def test_consensus_evaluator_edge_cases() -> None:
    """Test ConsensusEvaluator with no valid values and failed branch evaluation."""
    evaluator = ConsensusEvaluator(target_channel="answer")

    # Fit with no valid results
    evaluator.fit([])
    assert evaluator._consensus_value is None

    # Evaluate failed branch
    fail_res = BranchResult(branch_name="b_fail", error=ValueError("Failed"))
    res = evaluator.evaluate(fail_res)
    assert res.score == 0.0

    # Evaluate diverging branch
    evaluator.fit([BranchResult(branch_name="b_good", output={"answer": "42"})])
    diverge_res = BranchResult(branch_name="b_div", output={"answer": "100"})
    res_div = evaluator.evaluate(diverge_res)
    assert res_div.score == 0.0


def test_llm_judge_evaluator_parsing_and_errors() -> None:
    """Test LLMJudgeEvaluator error handling, unparseable floats, and failed branch."""
    # Failed branch
    judge = LLMJudgeEvaluator(judge_fn=lambda p: "SCORE: 0.9\nREASONING: Good")
    fail_res = BranchResult(branch_name="b1", error=RuntimeError("Crash"))
    assert judge.evaluate(fail_res).score == 0.0

    # Judge raising exception
    def broken_judge(prompt: str) -> str:
        _ = prompt
        msg = "LLM API Down"
        raise ConnectionError(msg)

    broken_eval = LLMJudgeEvaluator(judge_fn=broken_judge)
    ok_res = BranchResult(branch_name="b2", output={"text": "hello"})
    res_err = broken_eval.evaluate(ok_res)
    assert res_err.score == 0.0
    assert "LLM API Down" in str(res_err.reasoning)

    # Unparseable score float in response (fallback to 0.5 default)
    fallback_judge = LLMJudgeEvaluator(
        judge_fn=lambda p: "SCORE: not_a_number\nREASONING: Unknown score"
    )
    res_parsed = fallback_judge.evaluate(ok_res)
    assert res_parsed.score == 0.5

    # Multi-line reasoning without keyword
    multiline_judge = LLMJudgeEvaluator(
        judge_fn=lambda p: (
            "SCORE: 0.95\n"
            "This output is accurate and concise.\n"
            "Verified against rubric."
        )
    )
    res_multi = multiline_judge.evaluate(ok_res)
    assert res_multi.score == 0.95
    assert res_multi.reasoning is not None
    assert "accurate and concise" in res_multi.reasoning


# ==============================================================================
# 3. Speculative Node Decorator Edge Cases
# ==============================================================================


def test_speculative_node_config_dataclass() -> None:
    """Verify SpeculativeNodeConfig dataclass properties."""
    evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)
    config = SpeculativeNodeConfig(
        tools=[],
        evaluator=evaluator,
        prune_discarded=False,
        max_workers=4,
        fallback_on_all_fail=False,
    )
    assert config.max_workers == 4
    assert not config.prune_discarded
    assert not config.fallback_on_all_fail


def test_async_speculative_node_with_sync_tool() -> None:
    """Verify async @speculative_node executes sync tools in executor cleanly."""

    async def _run() -> None:
        evaluator = HeuristicEvaluator(
            scorer=lambda r: float(r.output.get("val", 0) if r.output else 0)
        )

        def sync_tool(state: dict[str, Any]) -> dict[str, Any]:
            return {"val": state.get("val", 0) + 10}

        async def async_tool(state: dict[str, Any]) -> dict[str, Any]:
            await asyncio.sleep(0.001)
            return {"val": state.get("val", 0) + 20}

        @speculative_node(
            tools=[sync_tool, async_tool],
            evaluator=evaluator,
        )
        async def async_node(state: dict[str, Any]) -> dict[str, Any]:
            return {"val": 0}

        result = await async_node({"val": 5})
        assert result["val"] == 25  # async_tool wins with 25
        meta = result.get("__speculative_meta__")
        assert meta is not None
        assert meta.winner_tool == "async_tool"

    asyncio.run(_run())


def test_async_speculative_node_all_fail_no_fallback() -> None:
    """Verify async @speculative_node when fallback_on_all_fail=False."""

    async def _run() -> None:
        evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)

        async def broken_async_tool(state: dict[str, Any]) -> dict[str, Any]:
            _ = state
            msg = "Async tool exploded"
            raise RuntimeError(msg)

        @speculative_node(
            tools=[broken_async_tool],
            evaluator=evaluator,
            fallback_on_all_fail=False,
        )
        async def fallback_node(state: dict[str, Any]) -> dict[str, Any]:
            return {"val": 999}

        result = await fallback_node({"val": 1})
        meta = result.get("__speculative_meta__")
        assert meta is not None
        assert not meta.used_fallback
        assert "val" not in result  # Fallback was not called

    asyncio.run(_run())


def test_sync_speculative_node_all_fail_no_fallback() -> None:
    """Verify sync @speculative_node when fallback_on_all_fail=False."""
    evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)

    def broken_sync_tool(state: dict[str, Any]) -> dict[str, Any]:
        _ = state
        msg = "Sync tool exploded"
        raise RuntimeError(msg)

    @speculative_node(
        tools=[broken_sync_tool],
        evaluator=evaluator,
        fallback_on_all_fail=False,
    )
    def fallback_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"val": 999}

    result = fallback_node({"val": 1})
    meta = result.get("__speculative_meta__")
    assert meta is not None
    assert not meta.used_fallback
    assert "val" not in result


# ==============================================================================
# 4. Beam Search Runner Edge Cases
# ==============================================================================


def test_beam_search_expand_fn_exception_resilience() -> None:
    """Verify BeamSearchRunner handles exceptions inside expand_fn gracefully."""
    saver = RiftCheckpointSaver()
    builder = StateGraph(SimpleState)

    def dummy_node(state: SimpleState) -> dict[str, Any]:
        return {"val": state["val"] + 1}

    builder.add_node("step", dummy_node)
    builder.add_edge(START, "step")
    graph = builder.compile(checkpointer=saver)

    config: RunnableConfig = {
        "configurable": {"thread_id": "beam-exc-thread", "checkpoint_ns": ""}
    }
    graph.invoke({"val": 0, "text": "start"}, config=config)

    def faulty_expand(beam: BranchResult) -> list[BranchSpec]:
        _ = beam
        msg = "Expansion crashed"
        raise ValueError(msg)

    runner = BeamSearchRunner(
        saver=saver,
        config=BeamSearchConfig(beam_width=2, branch_factor=2, max_depth=3),
    )
    evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)

    # Should not crash, should terminate gracefully at depth 0
    search_result = runner.run(
        graph=graph,
        initial_config=config,
        expand_fn=faulty_expand,
        evaluator=evaluator,
    )
    assert search_result.total_candidates_explored == 0
    assert search_result.depth_reached == 0


def test_beam_search_zero_candidates_early_stop() -> None:
    """Verify BeamSearchRunner terminates when expand_fn returns an empty list."""
    saver = RiftCheckpointSaver()
    builder = StateGraph(SimpleState)
    builder.add_node("step", lambda state: {"val": state["val"] + 1})
    builder.add_edge(START, "step")
    graph = builder.compile(checkpointer=saver)

    config: RunnableConfig = {
        "configurable": {"thread_id": "beam-empty-thread", "checkpoint_ns": ""}
    }
    graph.invoke({"val": 0, "text": "start"}, config=config)

    runner = BeamSearchRunner(
        saver=saver,
        config=BeamSearchConfig(beam_width=2, branch_factor=2, max_depth=5),
    )
    evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)

    search_result = runner.run(
        graph=graph,
        initial_config=config,
        expand_fn=lambda beam: [],
        evaluator=evaluator,
    )
    assert search_result.depth_reached == 0
    assert search_result.total_candidates_explored == 0


def test_async_beam_search_zero_candidates_early_stop() -> None:
    """Verify async arun terminates when expand_fn returns an empty list."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        builder = StateGraph(SimpleState)
        builder.add_node("step", lambda state: {"val": state["val"] + 1})
        builder.add_edge(START, "step")
        graph = builder.compile(checkpointer=saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": "beam-async-empty-thread",
                "checkpoint_ns": "",
            }
        }
        await graph.ainvoke({"val": 0, "text": "start"}, config=config)

        runner = BeamSearchRunner(
            saver=saver,
            config=BeamSearchConfig(beam_width=2, branch_factor=2, max_depth=5),
        )
        evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)

        search_result = await runner.arun(
            graph=graph,
            initial_config=config,
            expand_fn=lambda beam: [],
            evaluator=evaluator,
        )
        assert search_result.depth_reached == 0
        assert search_result.total_candidates_explored == 0

    asyncio.run(_run())


def test_beam_search_max_depth_zero() -> None:
    """Verify BeamSearchRunner with max_depth=0 returns seed result immediately."""
    saver = RiftCheckpointSaver()
    builder = StateGraph(SimpleState)
    builder.add_node("step", lambda state: {"val": state["val"] + 1})
    builder.add_edge(START, "step")
    graph = builder.compile(checkpointer=saver)

    config: RunnableConfig = {
        "configurable": {"thread_id": "beam-zero-depth", "checkpoint_ns": ""}
    }
    graph.invoke({"val": 10, "text": "start"}, config=config)

    runner = BeamSearchRunner(
        saver=saver,
        config=BeamSearchConfig(max_depth=0),
        evaluator=HeuristicEvaluator(scorer=lambda r: 5.0),
    )
    search_result = runner.run(
        graph=graph,
        initial_config=config,
        expand_fn=lambda b: [BranchSpec(name="c1", input_data={"val": 1})],
    )
    assert search_result.depth_reached == 0
    assert search_result.total_candidates_explored == 0


def test_async_beam_search_max_depth_zero() -> None:
    """Verify async BeamSearchRunner max_depth=0 returns seed result."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        builder = StateGraph(SimpleState)
        builder.add_node("step", lambda state: {"val": state["val"] + 1})
        builder.add_edge(START, "step")
        graph = builder.compile(checkpointer=saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": "beam-async-zero-depth",
                "checkpoint_ns": "",
            }
        }
        await graph.ainvoke({"val": 10, "text": "start"}, config=config)

        runner = BeamSearchRunner(
            saver=saver,
            config=BeamSearchConfig(max_depth=0),
            evaluator=HeuristicEvaluator(scorer=lambda r: 5.0),
        )
        search_result = await runner.arun(
            graph=graph,
            initial_config=config,
            expand_fn=lambda b: [BranchSpec(name="c1", input_data={"val": 1})],
        )
        assert search_result.depth_reached == 0
        assert search_result.total_candidates_explored == 0

    asyncio.run(_run())


# ==============================================================================
# 5. Multiverse Resolver & Visualization Edge Cases
# ==============================================================================


def test_multiverse_resolver_select_winner_empty_raises() -> None:
    """Verify select_winner raises ValueError on empty results."""
    saver = RiftCheckpointSaver()
    resolver = MultiverseResolver(saver=saver)
    with pytest.raises(ValueError, match="Cannot select winner from empty"):
        resolver.select_winner({}, {})


def test_plot_multiverse_with_output_path(tmp_path: Path) -> None:
    """Verify plot_multiverse executes with output_path."""
    saver = RiftCheckpointSaver()
    thread_id = "plot-thread"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    chk, meta = _make_chk("cp-plot-1", {"val": 1})
    saver.put(config, chk, meta, {"val": 1})

    plot_file = str(tmp_path / "plot.png")
    plot_multiverse(saver, thread_id, output_path=plot_file)

    # Test error fallback when savefig fails
    invalid_plot_file = "/non_existent_directory_12345/unwritable/plot.png"
    plot_multiverse(saver, thread_id, output_path=invalid_plot_file)


def test_base_saver_aget_tuple_and_get_tuple_missing() -> None:
    """Test BaseRiftSaver aget_tuple and missing checkpoint branches."""

    async def _run() -> None:
        saver = RiftCheckpointSaver()
        thread_id = "missing-cp-thread"
        cfg: RunnableConfig = {
            "configurable": {"thread_id": thread_id, "checkpoint_ns": "custom"}
        }
        # Non-existent storage
        assert saver.get_tuple(cfg) is None
        assert await saver.aget_tuple(cfg) is None

        # Add entry
        chk, meta = _make_chk("cp-1", {"val": 1})
        saver.put(cfg, chk, meta, {"val": 1})

        # Get latest without specifying checkpoint_id
        latest = saver.get_tuple(cfg)
        assert latest is not None
        assert latest.config["configurable"]["checkpoint_id"] == "cp-1"

        # Missing target_id
        missing_id_cfg: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "custom",
                "checkpoint_id": "non-existent-id",
            }
        }
        assert saver.get_tuple(missing_id_cfg) is None

        # Test base adelete_thread
        await saver.adelete_thread(thread_id)
        assert not saver.get_session_storage(thread_id)

    asyncio.run(_run())


def test_async_and_sync_saver_list_with_namespaces() -> None:
    """Test list and alist across multiple namespaces."""

    async def _run() -> None:
        # Sync saver list
        sync_saver = RiftCheckpointSaver()
        thread_id = "multi-ns-thread"
        for ns in ["ns1", "ns2"]:
            cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": f"cp-{ns}",
                }
            }
            chk, meta = _make_chk(f"cp-{ns}", {"ns": ns})
            sync_saver.put(cfg, chk, meta, {"ns": 1})

        # List all namespaces across threads
        all_cps = list(sync_saver.list(None))
        assert len(all_cps) >= 2

        # List filtered namespace
        ns1_cps = list(
            sync_saver.list(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": "ns1"}}
            )
        )
        assert len(ns1_cps) == 1

        # Async saver alist
        async_saver = AsyncRiftCheckpointSaver()
        a_thread_id = "async-multi-ns-thread"
        for ns in ["ans1", "ans2"]:
            a_cfg: RunnableConfig = {
                "configurable": {
                    "thread_id": a_thread_id,
                    "checkpoint_ns": ns,
                    "checkpoint_id": f"cp-{ns}",
                }
            }
            a_chk, a_meta = _make_chk(f"cp-{ns}", {"ns": ns})
            await async_saver.aput(a_cfg, a_chk, a_meta, {"ns": 1})

        all_a_cps = [cp async for cp in async_saver.alist(None)]
        assert len(all_a_cps) >= 2

        filtered_a_cps = [
            cp
            async for cp in async_saver.alist(
                {"configurable": {"thread_id": a_thread_id, "checkpoint_ns": "ans1"}}
            )
        ]
        assert len(filtered_a_cps) == 1

    asyncio.run(_run())


def test_speculative_tools_error_logging() -> None:
    """Test speculative tool error logging branches in sync and async runners."""

    def exploding_sync_tool(state: dict[str, Any]) -> dict[str, Any]:
        _ = state
        msg = "Sync explosion"
        raise ValueError(msg)

    def good_sync_tool(state: dict[str, Any]) -> dict[str, Any]:
        return {"val": 100}

    evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)

    @speculative_node(tools=[exploding_sync_tool, good_sync_tool], evaluator=evaluator)
    def sync_spec_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"val": 0}

    res_sync = sync_spec_node({"val": 0})
    assert res_sync["val"] == 100
    assert res_sync["__speculative_meta__"].winner_tool == "good_sync_tool"
    assert not res_sync["__speculative_meta__"].all_succeeded

    async def exploding_async_tool(state: dict[str, Any]) -> dict[str, Any]:
        _ = state
        await asyncio.sleep(0.001)
        msg = "Async explosion"
        raise ValueError(msg)

    async def good_async_tool(state: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.001)
        return {"val": 200}

    @speculative_node(
        tools=[exploding_async_tool, good_async_tool], evaluator=evaluator
    )
    async def async_spec_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"val": 0}

    async def _run() -> None:
        res_async = await async_spec_node({"val": 0})
        assert res_async["val"] == 200
        assert res_async["__speculative_meta__"].winner_tool == "good_async_tool"
        assert not res_async["__speculative_meta__"].all_succeeded

    asyncio.run(_run())


def test_beam_search_missing_evaluator_error() -> None:
    """Verify BeamSearchRunner raises ValueError if no evaluator is provided."""
    saver = RiftCheckpointSaver()
    builder = StateGraph(SimpleState)
    builder.add_node("step", lambda state: {"val": state["val"] + 1})
    builder.add_edge(START, "step")
    graph = builder.compile(checkpointer=saver)

    config: RunnableConfig = {
        "configurable": {"thread_id": "no-eval-thread", "checkpoint_ns": ""}
    }
    graph.invoke({"val": 0, "text": "start"}, config=config)

    runner = BeamSearchRunner(saver=saver)
    with pytest.raises(ValueError, match="An evaluator must be provided"):
        runner.run(
            graph=graph,
            initial_config=config,
            expand_fn=lambda b: [BranchSpec(name="c1", input_data={"val": 1})],
        )

    async def _run() -> None:
        with pytest.raises(ValueError, match="An evaluator must be provided"):
            await runner.arun(
                graph=graph,
                initial_config=config,
                expand_fn=lambda b: [BranchSpec(name="c1", input_data={"val": 1})],
            )

    asyncio.run(_run())


def test_beam_search_prune_discarded_and_no_survivors() -> None:
    """Verify BeamSearchRunner with prune_discarded and candidate pruning."""
    saver = RiftCheckpointSaver()
    builder = StateGraph(SimpleState)
    builder.add_node("step", lambda state: {"val": state["val"] + 1})
    builder.add_edge(START, "step")
    graph = builder.compile(checkpointer=saver)

    config: RunnableConfig = {
        "configurable": {"thread_id": "prune-beam-thread", "checkpoint_ns": ""}
    }
    graph.invoke({"val": 0, "text": "start"}, config=config)

    runner = BeamSearchRunner(
        saver=saver,
        config=BeamSearchConfig(
            beam_width=1, branch_factor=2, max_depth=2, prune_discarded=True
        ),
        evaluator=HeuristicEvaluator(scorer=lambda r: 1.0),
    )

    res = runner.run(
        graph=graph,
        initial_config=config,
        expand_fn=lambda b: [
            BranchSpec(name="b1", input_data={"val": 1}),
            BranchSpec(name="b2", input_data={"val": 2}),
        ],
    )
    assert res.winner is not None
    assert res.depth_reached == 2

    async def _run() -> None:
        async_saver = AsyncRiftCheckpointSaver()
        a_graph = builder.compile(checkpointer=async_saver)
        a_cfg: RunnableConfig = {
            "configurable": {"thread_id": "a-prune-beam-thread", "checkpoint_ns": ""}
        }
        await a_graph.ainvoke({"val": 0, "text": "start"}, config=a_cfg)

        a_runner = BeamSearchRunner(
            saver=async_saver,
            config=BeamSearchConfig(
                beam_width=1, branch_factor=2, max_depth=2, prune_discarded=True
            ),
            evaluator=HeuristicEvaluator(scorer=lambda r: 1.0),
        )
        a_res = await a_runner.arun(
            graph=a_graph,
            initial_config=a_cfg,
            expand_fn=lambda b: [
                BranchSpec(name="b1", input_data={"val": 1}),
                BranchSpec(name="b2", input_data={"val": 2}),
            ],
        )
        assert a_res.winner is not None
        assert a_res.depth_reached == 2

    asyncio.run(_run())


def test_speculative_node_winner_fallback_when_evaluator_favors_failed() -> None:
    """Verify fallback to successful tool when evaluator selects a failed tool."""

    def good_tool(state: dict[str, Any]) -> dict[str, Any]:
        return {"val": 10}

    def bad_tool(state: dict[str, Any]) -> dict[str, Any]:
        _ = state
        msg = "Error in tool"
        raise ValueError(msg)

    # Scorer intentionally gives failed branch high score
    evaluator = HeuristicEvaluator(scorer=lambda r: 10.0 if not r.is_success else 1.0)

    @speculative_node(tools=[good_tool, bad_tool], evaluator=evaluator)
    def sync_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"val": 0}

    res_sync = sync_node({"val": 1})
    assert res_sync["val"] == 10
    assert res_sync["__speculative_meta__"].winner_tool == "good_tool"

    async def a_good_tool(state: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.001)
        return {"val": 20}

    async def a_bad_tool(state: dict[str, Any]) -> dict[str, Any]:
        _ = state
        await asyncio.sleep(0.001)
        msg = "Async error in tool"
        raise ValueError(msg)

    @speculative_node(tools=[a_good_tool, a_bad_tool], evaluator=evaluator)
    async def async_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"val": 0}

    async def _run() -> None:
        res_async = await async_node({"val": 1})
        assert res_async["val"] == 20
        assert res_async["__speculative_meta__"].winner_tool == "a_good_tool"

    asyncio.run(_run())


def test_async_speculative_node_sync_fn_fallback() -> None:
    """Verify async @speculative_node fallback works when all async tools fail."""

    async def failing_tool(state: dict[str, Any]) -> dict[str, Any]:
        _ = state
        msg = "Failure"
        raise RuntimeError(msg)

    evaluator = HeuristicEvaluator(scorer=lambda r: 1.0)

    @speculative_node(
        tools=[failing_tool], evaluator=evaluator, fallback_on_all_fail=True
    )
    async def async_node(state: dict[str, Any]) -> dict[str, Any]:
        return {"val": 42}

    async def _run() -> None:
        res = await async_node({"val": 0})
        assert res["val"] == 42
        assert res["__speculative_meta__"].used_fallback

    asyncio.run(_run())


def test_async_saver_synchronous_wrappers() -> None:
    """Verify synchronous wrapper methods on AsyncRiftCheckpointSaver."""
    saver = AsyncRiftCheckpointSaver()
    thread_id = "sync-wrapper-thread"
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "cp-sync-1",
        }
    }

    # put
    chk, meta = _make_chk("cp-sync-1", {"val": 50})
    res_cfg = saver.put(config, chk, meta, {"val": 1})
    assert res_cfg["configurable"]["checkpoint_id"] == "cp-sync-1"

    # put_writes (and duplicate avoidance)
    saver.put_writes(config, [("channel", "v1")], "task-1")
    saver.put_writes(config, [("channel", "v1")], "task-1")

    # get_tuple
    tup = saver.get_tuple(config)
    assert tup is not None
    assert tup.checkpoint["channel_values"]["val"] == 50

    # get_tuple missing
    missing_cfg: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "non-existent",
        }
    }
    assert saver.get_tuple(missing_cfg) is None

    # list with limit
    items = list(saver.list(config, limit=1))
    assert len(items) == 1

    # delete_thread
    saver.delete_thread(thread_id)
    assert not saver.get_session_storage(thread_id)


def test_branch_manager_inspect_and_delete() -> None:
    """Verify BranchManager list_branches, get_branch_info, and delete_branch."""
    saver = RiftCheckpointSaver()
    thread_id = "bm-test-thread"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    chk, meta = _make_chk("cp-seed", {"val": 1})
    saver.put(config, chk, meta, {"val": 1})

    bm = BranchManager(saver)
    bm.create_branch(thread_id, "cand_a")
    bm.create_branch(thread_id, "cand_b")

    branches = bm.list_branches(thread_id)
    assert "cand_a" in branches
    assert "cand_b" in branches

    info = bm.get_branch_info(thread_id, "cand_a")
    assert info is not None
    assert info.branch_name == "cand_a"

    bm.delete_branch(thread_id, "cand_a")
    assert bm.get_branch_info(thread_id, "cand_a") is None


def test_multiverse_runner_arun_with_explicit_checkpoint() -> None:
    """Verify RiftRunner arun_parallel_branches with explicit checkpoint_id."""

    async def _run() -> None:
        saver = AsyncRiftCheckpointSaver()
        builder = StateGraph(SimpleState)
        builder.add_node("step", lambda state: {"val": state["val"] + 1, "text": "ok"})
        builder.add_edge(START, "step")
        graph = builder.compile(checkpointer=saver)

        config: RunnableConfig = {
            "configurable": {
                "thread_id": "mv-explicit-cp",
                "checkpoint_ns": "",
            }
        }
        initial_out = await graph.ainvoke({"val": 10, "text": "seed"}, config=config)
        assert initial_out["val"] == 11

        latest_tuple = await saver.aget_tuple(config)
        assert latest_tuple is not None
        seeded_cp_id = latest_tuple.config["configurable"]["checkpoint_id"]

        explicit_config: RunnableConfig = {
            "configurable": {
                "thread_id": "mv-explicit-cp",
                "checkpoint_ns": "",
                "checkpoint_id": seeded_cp_id,
            }
        }

        mv_runner = RiftRunner(saver=saver)
        results = await mv_runner.arun_parallel_branches(
            graph=graph,
            initial_config=explicit_config,
            branch_specs=[
                BranchSpec(name="b1", input_data={"val": 20, "text": "branch"})
            ],
        )
        assert "b1" in results
        assert results["b1"].output is not None
        assert results["b1"].output["val"] == 21

    asyncio.run(_run())


def test_base_rift_saver_direct_methods() -> None:
    """Verify BaseRiftSaver default get_tuple, aget_tuple, and delete methods."""
    saver = RiftCheckpointSaver()
    thread_id = "base-saver-test-thread"
    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": "cp-b-1",
        }
    }

    chk, meta = _make_chk("cp-b-1", {"val": 123})
    saver.put(config, chk, meta, {"val": 1})
    saver.put_writes(config, [("channel", "w1")], "task-1")

    # Base get_tuple
    tup = BaseRiftSaver.get_tuple(saver, config)
    assert tup is not None
    assert tup.checkpoint["channel_values"]["val"] == 123

    # Base get_tuple latest
    latest_tup = BaseRiftSaver.get_tuple(
        saver, {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    assert latest_tup is not None

    # Base delete_thread purging writes and blobs
    BaseRiftSaver.delete_thread(saver, thread_id)
    assert not saver.get_session_storage(thread_id)

    # Base aget_tuple & adelete_thread
    async def _run() -> None:
        cfg2: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
                "checkpoint_id": "cp-b-2",
            }
        }
        chk2, meta2 = _make_chk("cp-b-2", {"val": 456})
        saver.put(cfg2, chk2, meta2, {"val": 1})
        a_tup = await BaseRiftSaver.aget_tuple(saver, cfg2)
        assert a_tup is not None
        assert a_tup.checkpoint["channel_values"]["val"] == 456
        await BaseRiftSaver.adelete_thread(saver, thread_id)
        assert not saver.get_session_storage(thread_id)

    asyncio.run(_run())


def test_resolver_evaluate_and_aevaluate_with_consensus() -> None:
    """Verify MultiverseResolver evaluation with ConsensusEvaluator."""
    saver = RiftCheckpointSaver()
    resolver = MultiverseResolver(saver=saver)
    evaluator = ConsensusEvaluator(target_channel="answer")

    results = {
        "b1": BranchResult(branch_name="b1", output={"answer": "alpha"}),
        "b2": BranchResult(branch_name="b2", output={"answer": "alpha"}),
        "b3": BranchResult(branch_name="b3", output={"answer": "beta"}),
    }

    # Synchronous evaluation
    sync_scores = resolver.evaluate_branches(results, evaluator)
    assert sync_scores["b1"].score == pytest.approx(2 / 3)
    assert sync_scores["b2"].score == pytest.approx(2 / 3)
    assert sync_scores["b3"].score == 0.0

    async def _run() -> None:
        eval_results = await resolver.aevaluate_branches(results, evaluator)
        assert eval_results["b1"].score == pytest.approx(2 / 3)
        assert eval_results["b2"].score == pytest.approx(2 / 3)
        assert eval_results["b3"].score == 0.0

        winner = resolver.select_winner(eval_results, results)
        assert winner in ["b1", "b2"]

    asyncio.run(_run())
