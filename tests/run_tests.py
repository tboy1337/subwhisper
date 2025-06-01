#!/usr/bin/env python3
"""
Test runner for SubWhisper
This script runs all the tests for the SubWhisper application
"""

import os
import sys
import unittest
import argparse

# Add parent directory to sys.path to ensure we can import subwhisper
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_tests(verbose=False):
    """Run all the tests for SubWhisper"""
    # Discover and run tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(os.path.abspath(__file__)), pattern="test_*.py")
    
    # Configure test runner
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    
    print("=" * 70)
    print("Running SubWhisper Tests")
    print("=" * 70)
    
    # Run tests
    result = runner.run(test_suite)
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SubWhisper tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    # Run tests and exit with appropriate code
    sys.exit(run_tests(args.verbose)) 