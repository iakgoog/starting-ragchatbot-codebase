#!/usr/bin/env python3
"""
Quick formatting script - formats code with black and isort.
Usage: uv run format.py
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"[RUNNING] {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"[PASSED] {description}")
        if result.stdout.strip():
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
    print("Running code formatting tools...")
    
    all_passed = True
    
    # Run isort
    success = run_command(
        "uv run isort backend/ main.py scripts/ format.py",
        "Import sorting (isort)"
    )
    all_passed &= success
    
    # Run black
    success = run_command(
        "uv run black backend/ main.py scripts/ format.py",
        "Code formatting (black)"
    )
    all_passed &= success
    
    print("\n" + "="*50)
    if all_passed:
        print("[SUCCESS] Code formatting completed!")
        sys.exit(0)
    else:
        print("[ERROR] Some formatting failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()