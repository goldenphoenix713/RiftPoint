# AGENTS.md

## Project Overview

**RiftPoint** is a high-performance multiversal checkpointer and state management engine for [LangGraph](https://github.com/langchain-ai/langgraph), powered by the Rust-backed [**`janus-tachyon-rs`**](https://github.com/goldenphoenix713/janus) (`Tachyon-RS`) core.

### Core Mission

In standard agent architectures, exploring alternate reasoning branches, tool call variations, or speculative futures requires expensive state duplication or complex tree orchestration. RiftPoint intercepts an agent's execution DAG to enable:

- **Multiversal Branching:** Instant fork of agent state into parallel candidate realities with near-zero memory copy penalty.
- **Speculative Tool Calling:** Concurrent execution of divergent tool-use strategies across distinct timeline branches.
- **Multiverse Collapse:** Scoring candidate timelines against evaluation criteria, committing winning trajectories, and pruning discarded branches.
- **Full State Travel & Tracing:** Interactive time travel, undo/redo, and DAG visualization (Mermaid / Matplotlib) backed by Janus.

---

## Repository Architecture

```text
RiftPoint/
├── .github/
│   └── workflows/
│       └── ci.yml               # CI Pipeline (Python 3.11 & 3.12, Lint, Typecheck, Radon, Pytest)
├── scripts/
│   └── check_radon.py           # Radon Cyclomatic Complexity & Maintainability Index enforcer
├── src/
│   └── riftpoint/
│       ├── __init__.py          # Package exports and versioning
│       └── py.typed             # PEP 561 typing marker
├── tests/
│   └── test_init.py             # Package initialization & smoke tests
├── .markdownlintrc              # Markdown linting configuration
├── .pre-commit-config.yaml      # Git pre-commit hooks configuration
├── AGENTS.md                    # Agent guidelines & project configuration (this file)
├── pyproject.toml               # Project metadata, dependencies, and tool settings
└── README.md                    # Project README
```

---

## Tech Stack & Tooling

| Component | Tool / Technology | Version / Configuration |
| :--- | :--- | :--- |
| **Language** | Python | `>= 3.11` |
| **Package Manager** | `uv` | Dependency resolution & workspace runner |
| **Build Backend** | `uv_build` | `>= 0.10.0, < 0.13.0` |
| **Core Engine** | `janus-tachyon-rs` | Rust-backed multiversal state engine |
| **Agent Framework** | `langgraph` / `langchain-core` | State graph & checkpointer integration |
| **Linter & Formatter** | `ruff` | Comprehensive ruleset (`select = ["ALL"]`, line length `88`) |
| **Type Checker** | `mypy` | `strict = true` (Zero untyped defs/calls) |
| **Code Metrics** | `radon` | Grade B or better (CC $\le 10$, MI $\ge 10$) |
| **Test Suite** | `pytest` + `pytest-cov` | Minimum code coverage requirement $\ge 70\%$ |
| **Git Hooks** | `pre-commit` | Automated pre-commit verification |
| **CI / CD** | GitHub Actions | Matrix validation on Ubuntu (Python 3.11, 3.12) |

---

## Developer & Agent Commands

All commands are executed using `uv` within the workspace:

### 1. Environment & Dependencies

```bash
# Sync all dependencies including dev groups
uv sync --all-groups
```

### 2. Testing & Coverage

```bash
# Run full test suite with coverage enforcement (>= 70%)
uv run pytest

# Run tests with detailed verbose output
uv run pytest -v -s
```

### 3. Linting & Formatting

```bash
# Run Ruff lint checks
uv run ruff check

# Auto-fix Ruff lint violations
uv run ruff check --fix

# Verify formatting without modifying files
uv run ruff format --check

# Format code with Ruff
uv run ruff format
```

### 4. Type Checking

```bash
# Run strict Mypy type check across source and test files
uv run mypy src tests
```

### 5. Code Quality & Complexity Metrics

```bash
# Verify Cyclomatic Complexity & Maintainability Index (Grade B or better)
uv run python scripts/check_radon.py src
```

### 6. Markdown Linting

```bash
# Run markdownlint against all markdown files
markdownlint "**/*.md" --ignore ".venv/**" --ignore "node_modules/**"

# Automatically fix common markdown formatting issues
markdownlint "**/*.md" --fix --ignore ".venv/**" --ignore "node_modules/**"
```

### 7. Pre-commit Hooks

```bash
# Run all pre-commit hooks manually against all files
pre-commit run --all-files
```

---

## Coding Standards & Agent Constraints

Agents and contributors modifying this codebase must adhere to the following rules:

1. **Strict Type Safety:**
   - All modules, functions, classes, and methods must have complete type annotations.
   - `mypy` strict mode is enabled. Avoid `Any` where possible; use generics, protocols, or explicit union types.

2. **Radon Metric Enforcement:**
   - Functions and methods must maintain **Grade B or better** Cyclomatic Complexity ($CC \le 10$).
   - Modules must maintain **Grade B or better** Maintainability Index ($MI \ge 10$).
   - Avoid monolithic functions; break down complex logic into composable helpers.

3. **Coverage Standard:**
   - Every new feature or bugfix must be accompanied by unit tests.
   - Total test coverage must never drop below **70%** (enforced by `pytest-cov` and CI).

4. **Formatting & Docstrings:**
   - Follow Google/PEP 257 docstring conventions.
   - Respect Ruff configuration (double quotes, 4-space indentation, 88 character line width).

5. **Markdown Quality:**
   - Markdown documents must adhere to [`.markdownlintrc`](.markdownlintrc) rules with no lint errors or missing blank lines around lists/blocks.
