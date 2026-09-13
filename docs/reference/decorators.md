# Speculative Node Decorator

Declarative ergonomics for turning LangGraph nodes into concurrent speculative tool races.

## Overview

The `@speculative_node` decorator transforms any standard LangGraph node function
into a parallel race across multiple candidate tools or strategies. The runner
evaluates all candidate outputs concurrently, selects the highest-scoring candidate,
and collapses the node state to the winner.

## Decorator

::: riftpoint.decorators.speculative_node

## Configuration & Metadata

::: riftpoint.decorators.SpeculativeNodeConfig

::: riftpoint.decorators.SpeculativeRaceMeta
