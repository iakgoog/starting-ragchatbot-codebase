#!/usr/bin/env python3
"""
Quality check script that runs all code quality tools.
Usage: uv run scripts/quality_check.py [--fix]
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, fix_mode=False):
    """Run a command and return True if successful."""
    print(f"\n[RUNNING] {description}...")
    try:
        result = subprocess.run(
            cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(f"[PASSED] {description}")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] {description}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix issues where possible"
    )
    args = parser.parse_args()

    # Change to project root
    project_root = Path(__file__).parent.parent
    print(f"Running quality checks from: {project_root}")

    all_passed = True

    if args.fix:
        print("[FIX MODE] Running in fix mode - will auto-fix issues where possible")

        # Auto-fix with isort
        success = run_command(
            "uv run isort backend/ main.py scripts/", "Import sorting (isort) - fixing"
        )
        all_passed &= success

        # Auto-fix with black
        success = run_command(
            "uv run black backend/ main.py scripts/", "Code formatting (black) - fixing"
        )
        all_passed &= success
    else:
        # Check mode only
        success = run_command(
            "uv run isort --check-only --diff backend/ main.py scripts/",
            "Import sorting (isort) - check only",
        )
        all_passed &= success

        success = run_command(
            "uv run black --check --diff backend/ main.py scripts/",
            "Code formatting (black) - check only",
        )
        all_passed &= success

    # Always run these checks (no fix mode)
    success = run_command("uv run flake8 backend/ main.py scripts/", "Linting (flake8)")
    all_passed &= success

    success = run_command(
        "uv run mypy backend/ main.py scripts/", "Type checking (mypy)"
    )
    all_passed &= success

    # Run tests
    success = run_command("uv run pytest backend/tests/ -v", "Running tests")
    all_passed &= success

    print("\n" + "=" * 50)
    if all_passed:
        print("[SUCCESS] All quality checks passed!")
        sys.exit(0)
    else:
        print("[ERROR] Some quality checks failed. Please fix the issues above.")
        if not args.fix:
            print("[TIP] Try running with --fix to auto-fix formatting issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
