"""Radon metric enforcement script.

Checks Python files for Cyclomatic Complexity (CC) and Maintainability Index (MI)
and exits with code 1 if any metric fails to meet Grade B or better.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from radon.complexity import cc_rank, cc_visit
from radon.metrics import mi_rank, mi_visit

if TYPE_CHECKING:
    from collections.abc import Sequence

ALLOWED_RANKS = {"A", "B"}


def find_python_files(
    targets: Sequence[str],
    exclude_patterns: Sequence[str],
) -> list[Path]:
    """Find all python files under target paths excluding patterns."""
    files: list[Path] = []
    for target_str in targets:
        target = Path(target_str)
        if target.is_file() and target.suffix == ".py":
            files.append(target)
        elif target.is_dir():
            for p in target.rglob("*.py"):
                p_str = str(p)
                if any(ex in p_str for ex in exclude_patterns):
                    continue
                files.append(p)
    return sorted(set(files))


def check_cyclomatic_complexity(
    file_path: Path,
    source: str,
    max_rank: str = "B",
) -> list[str]:
    """Check cyclomatic complexity for all functions/classes in a file."""
    violations: list[str] = []
    try:
        blocks = cc_visit(source)
    except Exception as exc:  # noqa: BLE001
        violations.append(f"{file_path}: Failed to parse AST for CC check: {exc}")
        return violations

    for block in blocks:
        rank = cc_rank(block.complexity)
        if rank not in ALLOWED_RANKS and (
            max_rank == "B" or (max_rank == "A" and rank != "A")
        ):
            violations.append(
                f"{file_path}:{block.lineno} - {block.name} (line {block.lineno}) "
                f"has CC score {block.complexity} (Grade {rank}). "
                f"Required: Grade {max_rank} or better.",
            )
    return violations


def check_maintainability_index(
    file_path: Path,
    source: str,
    min_rank: str = "B",
) -> list[str]:
    """Check Maintainability Index (MI) score for a python file."""
    violations: list[str] = []
    try:
        mi_score = mi_visit(source, multi=True)
        rank = mi_rank(mi_score)
    except Exception as exc:  # noqa: BLE001
        violations.append(f"{file_path}: Failed to parse AST for MI check: {exc}")
        return violations

    if rank not in ALLOWED_RANKS and (
        min_rank == "B" or (min_rank == "A" and rank != "A")
    ):
        violations.append(
            f"{file_path} - Maintainability Index is {mi_score:.2f} (Grade {rank}). "
            f"Required: Grade {min_rank} or better.",
        )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    """Run complexity and maintainability validation across target files."""
    parser = argparse.ArgumentParser(
        description="Verify radon Cyclomatic Complexity and Maintainability Index.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=["src"],
        help="Directories or files to inspect (default: src)",
    )
    parser.add_argument(
        "--max-cc",
        default="B",
        choices=["A", "B"],
        help="Maximum allowed Cyclomatic Complexity rank (default: B)",
    )
    parser.add_argument(
        "--min-mi",
        default="B",
        choices=["A", "B"],
        help="Minimum allowed Maintainability Index rank (default: B)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["tests", ".venv", "__pycache__", "build", "dist"],
        help="Patterns to exclude from inspection",
    )

    args = parser.parse_args(argv)

    py_files = find_python_files(args.targets, args.exclude)
    if not py_files:
        print("No Python files found to inspect.")
        return 0

    cc_violations: list[str] = []
    mi_violations: list[str] = []

    for file_path in py_files:
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            print(f"Error reading {file_path}: {exc}", file=sys.stderr)
            return 1

        cc_violations.extend(
            check_cyclomatic_complexity(file_path, source, max_rank=args.max_cc),
        )
        mi_violations.extend(
            check_maintainability_index(file_path, source, min_rank=args.min_mi),
        )

    summary = (
        f"Inspected {len(py_files)} Python file(s) for Radon metrics "
        f"(CC <= {args.max_cc}, MI >= {args.min_mi})."
    )
    print(summary)

    if cc_violations or mi_violations:
        print("\n❌ Radon Metric Violations Found:\n", file=sys.stderr)
        if cc_violations:
            print("Cyclomatic Complexity (CC) Violations:", file=sys.stderr)
            for v in cc_violations:
                print(f"  • {v}", file=sys.stderr)
        if mi_violations:
            print("\nMaintainability Index (MI) Violations:", file=sys.stderr)
            for v in mi_violations:
                print(f"  • {v}", file=sys.stderr)
        return 1

    print("✅ All inspected files meet the required Radon metrics (Grade B or better).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
