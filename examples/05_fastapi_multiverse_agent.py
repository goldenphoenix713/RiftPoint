"""05_fastapi_multiverse_agent.py — Async Multiverse Agent Service Architecture.

Demonstrates:
1. High-throughput asynchronous agent serving with AsyncRiftCheckpointSaver.
2. Concurrent non-blocking speculative branch evaluation.
3. FastAPI endpoint blueprints for /query, /speculate, and /collapse.
4. Background timeline pruning and asynchronous state commitment.
5. Standalone simulated async client demonstrating end-to-end execution.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import (
    AsyncRiftCheckpointSaver,
    BranchSpec,
    HeuristicEvaluator,
    MultiverseResolver,
    RiftRunner,
)

try:
    from fastapi import FastAPI  # type: ignore[import-not-found]
    from pydantic import BaseModel  # type: ignore[import-not-found]

    HAS_FASTAPI = True
except ImportError:  # pragma: no cover
    HAS_FASTAPI = False

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

    from riftpoint.runner.multiverse import BranchResult


# 1. State Schema
class ServiceAgentState(TypedDict):
    """Agent state for async service."""

    messages: Annotated[list[Any], add_messages]
    request_id: str
    response: str
    confidence: float


# 2. Graph Node: Async Response Generation
async def async_agent_node(state: ServiceAgentState) -> dict[str, Any]:
    """Async agent reasoning node."""
    await asyncio.sleep(0.01)  # Simulate async I/O
    req_id = state.get("request_id", "req-unknown")
    last_msg = state["messages"][-1].content if state.get("messages") else ""
    return {
        "response": f"Processed '{last_msg}' for request {req_id}",
        "confidence": 0.95,
        "messages": [AIMessage(content=f"Agent completed processing for {req_id}")],
    }


# 3. Multiverse Agent Service Controller
class MultiverseAgentService:
    """Async controller managing LangGraph executions across parallel timelines."""

    def __init__(self) -> None:
        self.saver = AsyncRiftCheckpointSaver()
        self.runner = RiftRunner(saver=self.saver)
        self.resolver = MultiverseResolver(saver=self.saver)

        # Build and compile graph
        builder = StateGraph(ServiceAgentState)
        builder.add_node("agent", async_agent_node)
        builder.add_edge(START, "agent")
        self.graph = builder.compile(checkpointer=self.saver)

    async def handle_query(
        self,
        session_id: str,
        user_message: str,
        request_id: str,
    ) -> dict[str, Any]:
        """Execute canonical query turn."""
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id, "checkpoint_ns": ""}
        }
        input_state: ServiceAgentState = {
            "messages": [HumanMessage(content=user_message)],
            "request_id": request_id,
            "response": "",
            "confidence": 0.0,
        }
        return await self.graph.ainvoke(input_state, config=config)

    async def handle_speculate(
        self,
        session_id: str,
        candidate_prompts: list[tuple[str, str]],
    ) -> dict[str, BranchResult]:
        """Spawn and execute parallel speculative branches asynchronously."""
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id, "checkpoint_ns": ""}
        }
        branch_specs = [
            BranchSpec(
                name=branch_name,
                input_data={
                    "messages": [HumanMessage(content=prompt)],
                    "request_id": f"spec-{branch_name}",
                },
            )
            for branch_name, prompt in candidate_prompts
        ]
        return await self.runner.arun_parallel_branches(
            graph=self.graph,
            initial_config=config,
            branch_specs=branch_specs,
        )

    async def handle_collapse(
        self,
        session_id: str,
        results: dict[str, BranchResult],
    ) -> tuple[str, float]:
        """Evaluate candidate realities and collapse to the winning trajectory."""

        def score_fn(result: BranchResult) -> float:
            if not result.is_success or not result.output:
                return 0.0
            return float(result.output.get("confidence", 0.0))

        evaluator = HeuristicEvaluator(scorer=score_fn)
        summary = await self.resolver.acollapse(
            thread_id=session_id,
            results=results,
            evaluator=evaluator,
            prune_discarded=True,
        )
        winner_score = summary.scores[summary.winning_branch].score
        return summary.winning_branch, winner_score


# 4. FastAPI Blueprint (Optional Integration)
def create_fastapi_app() -> Any:
    """Create FastAPI application instance if FastAPI is installed."""
    if not HAS_FASTAPI:
        return None

    app = FastAPI(
        title="RiftPoint Multiverse Agent API",
        description="Async agent serving with Janus-backed multiverse state.",
        version="1.0.0",
    )
    service = MultiverseAgentService()

    class QueryRequest(BaseModel):  # type: ignore[misc]
        session_id: str
        message: str
        request_id: str = "req-01"

    class SpeculateRequest(BaseModel):  # type: ignore[misc]
        session_id: str
        candidates: dict[str, str]

    @app.post("/query")  # type: ignore[misc]
    async def query_endpoint(req: QueryRequest) -> dict[str, Any]:
        result = await service.handle_query(req.session_id, req.message, req.request_id)
        return {"status": "success", "data": result}

    @app.post("/speculate")  # type: ignore[misc]
    async def speculate_endpoint(req: SpeculateRequest) -> dict[str, Any]:
        prompts = list(req.candidates.items())
        branch_results = await service.handle_speculate(req.session_id, prompts)
        winner, score = await service.handle_collapse(req.session_id, branch_results)
        return {
            "status": "collapsed",
            "winning_branch": winner,
            "winner_score": score,
            "evaluated_branches": list(branch_results.keys()),
        }

    return app


# 5. Standalone Async Execution Simulation
async def run_async_simulation() -> None:
    """Run end-to-end async service simulation."""
    print("=" * 75)
    print("⚡ RiftPoint: Asynchronous Multiverse Agent Service")
    print("=" * 75)

    service = MultiverseAgentService()
    session_id = "async-service-demo-01"

    # Turn 1: Canonical query
    print("\n--- 1. Asynchronous Canonical Query ---")
    canonical_result = await service.handle_query(
        session_id=session_id,
        user_message="Analyze market trends for Q3",
        request_id="req-101",
    )
    print(f"Canonical Response:   {canonical_result['response']}")
    print(f"Canonical Confidence: {canonical_result['confidence']:.2f}")

    # Turn 2: Concurrent Speculative Racing across 3 strategies
    print("\n--- 2. Asynchronous Multiversal Speculation (3 Parallel Branches) ---")
    candidate_prompts = [
        ("deep_research", "Perform multi-source financial report synthesis"),
        ("realtime_news", "Fetch breaking sector news headlines"),
        ("quantitative", "Run Monte Carlo volatility simulation"),
    ]

    branch_results = await service.handle_speculate(
        session_id=session_id,
        candidate_prompts=candidate_prompts,
    )

    for name, res in branch_results.items():
        resp = res.output.get("response") if res.output else "Error"
        print(f"  Branch '{name}': success={res.is_success}, response='{resp}'")

    # Turn 3: Asynchronous Collapse
    print("\n--- 3. Asynchronous Multiverse Collapse ---")
    winner, winner_score = await service.handle_collapse(session_id, branch_results)
    print(f"Winning Strategy:     '{winner}' (score: {winner_score:.2f})")

    # Turn 4: Checkpoint DAG Verification
    print("\n--- 4. Active Janus DAG (Mermaid) ---")
    print(service.saver.visualize(session_id))

    print("\n✅ Asynchronous multiverse agent simulation completed successfully!")


def main() -> None:
    """Entry point for standalone execution."""
    asyncio.run(run_async_simulation())


if __name__ == "__main__":
    main()
