#!/usr/bin/env python3
"""
Development script for running code quality checks.
"""
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔍 {description}")
    print(f"Running: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} passed")
            return True
        else:
            print(f"❌ {description} failed")
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def main():
    """Run all quality checks."""
    print("🚀 Running code quality checks...")

    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    checks = [
        (["uv", "run", "black", "--check", "."], "Black formatting check"),
        (["uv", "run", "isort", "--check-only", "."], "Import sorting check"),
        (["uv", "run", "flake8", "."], "Flake8 linting"),
        (["uv", "run", "mypy", "backend/"], "MyPy type checking"),
    ]

    all_passed = True
    for command, description in checks:
        if not run_command(command, description):
            all_passed = False

    if all_passed:
        print("\n🎉 All quality checks passed!")
        sys.exit(0)
    else:
        print("\n💥 Some quality checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
