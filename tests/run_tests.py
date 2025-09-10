#!/usr/bin/env python3
"""
Test runner for SubWhisper - Pytest-based
This script runs all the pytest-based tests for the SubWhisper application
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(verbose=False, coverage=True, timeout=300):
    """Run all the pytest-based tests for SubWhisper"""
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent / "test_subwhisper.py"
    cmd.append(str(test_dir))
    
    # Add options
    if verbose:
        cmd.extend(["-v", "--tb=short"])
    
    if coverage:
        cmd.extend(["--cov=subwhisper", "--cov-report=term-missing", "--cov-report=html"])
    
    cmd.extend([f"--timeout={timeout}"])
    
    print("=" * 70)
    print("Running SubWhisper Pytest Tests")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SubWhisper pytest tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--timeout", type=int, default=300, help="Test timeout in seconds")
    args = parser.parse_args()
    
    # Run tests and exit with appropriate code
    sys.exit(run_tests(
        verbose=args.verbose,
        coverage=not args.no_coverage,
        timeout=args.timeout
    )) 