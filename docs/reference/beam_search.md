# Beam Search Runner

Multi-depth beam search orchestration engine for autonomous Tree-of-Thought
reasoning over LangGraph state graphs.

## Overview

The `BeamSearchRunner` iteratively expands, evaluates, selects, and prunes
candidate timelines at each depth level, composing existing RiftPoint
infrastructure (`RiftRunner`, `BranchManager`, evaluators) to enable
multi-step tree search in a single API call.

## Configuration

::: riftpoint.runner.beam.BeamSearchConfig

## Runner

::: riftpoint.runner.beam.BeamSearchRunner

## Result Types

::: riftpoint.runner.beam.BeamSearchResult

::: riftpoint.runner.beam.DepthSummary
