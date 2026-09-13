"""04_time_travel_debugger.py — DAG Time Travel, State Rewind, and Counterfactuals.

Demonstrates:
1. Stepping through a multi-turn agent workflow.
2. Inspecting the full historical lineage of checkpoints in Janus DAG.
3. Rewinding execution back to an earlier checkpoint in time.
4. Forking a counterfactual "what-if" branch from the historical state.
5. Comparing the divergent realities and visualizing the multiverse DAG.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages

from riftpoint import (
    BranchSpec,
    RiftCheckpointSaver,
    RiftRunner,
)

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


# 1. State Schema
class DebuggerAgentState(TypedDict):
    """Agent state for time travel debugging."""

    messages: Annotated[list[Any], add_messages]
    step_num: int
    variable_x: int
    decision: str


# 2. Graph Node: Simulates a state mutation step
def process_node(state: DebuggerAgentState) -> dict[str, Any]:
    """Process current state and apply mutations."""
    curr_step = state.get("step_num", 0) + 1
    curr_x = state.get("variable_x", 10)
    decision = state.get("decision", "default_path")

    # Apply arithmetic based on decision
    if decision == "aggressive_growth":
        new_x = curr_x * 3
    elif decision == "conservative_growth":
        new_x = curr_x + 5
    elif decision == "risky_experiment":
        new_x = curr_x * 10 - 50
    else:
        new_x = curr_x + 1

    msg = f"Step {curr_step}: applied '{decision}' → variable_x is now {new_x}"
    return {
        "step_num": curr_step,
        "variable_x": new_x,
        "messages": [AIMessage(content=msg)],
    }


def main() -> None:
    """Run time travel debugger demonstration."""
    print("=" * 75)
    print("⏳ RiftPoint: DAG Time Travel & Counterfactual Trajectory Debugger")
    print("=" * 75)

    # 3. Build & Compile Graph
    builder = StateGraph(DebuggerAgentState)
    builder.add_node("process", process_node)
    builder.add_edge(START, "process")

    saver = RiftCheckpointSaver()
    graph = builder.compile(checkpointer=saver)

    thread_id = "debug-session-01"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }

    # 4. Step 1-3: Run initial sequential turns
    print("\n--- 1. Initial Timeline Execution (Steps 1 to 3) ---")
    inputs = [
        {
            "decision": "conservative_growth",
            "variable_x": 10,
            "messages": [HumanMessage(content="Start")],
        },
        {"decision": "conservative_growth"},
        {"decision": "conservative_growth"},
    ]

    for turn_idx, turn_input in enumerate(inputs, 1):
        res = graph.invoke(turn_input, config=config)
        last_msg = res["messages"][-1].content
        print(f"Turn {turn_idx}: x = {res['variable_x']} | Last msg: {last_msg}")

    # 5. List Checkpoint History
    print("\n--- 2. Inspecting Checkpoint History in Janus DAG ---")
    checkpoints = list(saver.list(config))
    print(f"Total Checkpoints Recorded: {len(checkpoints)}")
    for idx, cp in enumerate(checkpoints):
        step = cp.metadata.get("step", "N/A")
        cp_id = cp.config["configurable"].get("checkpoint_id", "N/A")
        x_val = cp.checkpoint.get("channel_values", {}).get("variable_x", "N/A")
        print(f"  [{idx}] Checkpoint ID: {cp_id} | step: {step} | variable_x: {x_val}")

    # 6. Rewind to Step 1 Checkpoint (picking step 1 checkpoint)
    target_checkpoint = None
    for cp in checkpoints:
        if cp.metadata.get("step") == 1:
            target_checkpoint = cp
            break

    if not target_checkpoint:
        target_checkpoint = checkpoints[-1]

    rewind_cp_id = target_checkpoint.config["configurable"]["checkpoint_id"]
    rewind_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": rewind_cp_id,
        }
    }
    print(f"\n--- 3. Rewinding State to Checkpoint '{rewind_cp_id}' ---")
    rewound_state = saver.get_tuple(rewind_config)
    if rewound_state:
        print(f"Restored Channel Values: {rewound_state.checkpoint['channel_values']}")

    # 7. Spawn Counterfactual "What-If" Branches from Rewound State
    print("\n--- 4. Exploring Counterfactual Trajectories from Rewound State ---")
    runner = RiftRunner(saver=saver)
    counterfactual_results = runner.run_parallel_branches(
        graph=graph,
        initial_config=rewind_config,
        branch_specs=[
            BranchSpec(
                name="counterfactual_aggressive",
                input_data={"decision": "aggressive_growth"},
            ),
            BranchSpec(
                name="counterfactual_risky",
                input_data={"decision": "risky_experiment"},
            ),
        ],
    )

    for branch_name, res in counterfactual_results.items():
        if res.output:
            print(f"  Trajectory '{branch_name}':")
            print(f"    variable_x = {res.output.get('variable_x')}")
            print(f"    message    = {res.output.get('messages')[-1].content}")

    # 8. Render Multiversal DAG
    print("\n--- 5. Multiverse DAG Visualization (Mermaid) ---")
    print(saver.visualize(thread_id))

    print("\n✅ Time travel debugger completed successfully!")


if __name__ == "__main__":
    main()
