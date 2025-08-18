#!/usr/bin/env python3
"""
Development script for formatting code automatically.
"""
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n🔧 {description}")
    print(f"Running: {' '.join(command)}")

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout:
                print(result.stdout)
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
    """Format all code automatically."""
    print("🎨 Formatting code...")

    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    formatters = [
        (["uv", "run", "isort", "."], "Sorting imports"),
        (["uv", "run", "black", "."], "Formatting with Black"),
    ]

    all_passed = True
    for command, description in formatters:
        if not run_command(command, description):
            all_passed = False

    if all_passed:
        print("\n🎉 Code formatting completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some formatting operations failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
