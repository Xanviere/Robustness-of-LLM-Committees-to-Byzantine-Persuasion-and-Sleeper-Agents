import pytest
import asyncio
import os
import sys

# Ensure byzantine_mas root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from verifiers.numeric_verifier import verify_numeric, verify_aime, extract_numeric_scalar
from verifiers.math_verifier import verify_math, normalize_latex
from verifiers.code_sandbox import CodeSandbox

def test_extract_numeric_scalar():
    assert extract_numeric_scalar("72") == 72.0
    assert extract_numeric_scalar("The answer is $42.50 dollars.") == 42.50
    assert extract_numeric_scalar(r"\boxed{100}") == 100.0
    assert extract_numeric_scalar("The fraction is 3/4.") == 0.75
    assert extract_numeric_scalar("-15.25") == -15.25
    assert extract_numeric_scalar("No numbers here") is None

def test_verify_numeric():
    assert verify_numeric("72", "72") is True
    assert verify_numeric("Total: 72 clips", "72") is True
    assert verify_numeric("72.0", "72") is True
    assert verify_numeric("73", "72") is False

def test_verify_aime():
    assert verify_aime("0", "0") is True
    assert verify_aime("4", "4") is True
    assert verify_aime("The remainder is 4.", "4") is True
    assert verify_aime("5", "4") is False

def test_verify_math():
    # Direct equivalence
    assert verify_math(r"\frac{1}{2}", r"\frac{2}{4}") is True
    assert verify_math(r"x + y", r"y + x") is True
    assert verify_math(r"\sqrt{4}", "2") is True
    assert verify_math(r"2\pi", r"2 * \pi") is True
    assert verify_math(r"\boxed{x^2 - 1}", r"(x - 1)(x + 1)") is True
    assert verify_math(r"x + 1", r"x + 2") is False

def test_code_sandbox():
    code = "def add(a, b): return a + b"
    test_cases = "assert add(2, 3) == 5\nassert add(-1, 1) == 0"
    success = CodeSandbox.run_in_docker_sync(code, test_cases)
    assert success is True

    failing_test = "assert add(2, 2) == 5"
    fail_res = CodeSandbox.run_in_docker_sync(code, failing_test)
    assert fail_res is False

if __name__ == "__main__":
    test_extract_numeric_scalar()
    test_verify_numeric()
    test_verify_aime()
    test_verify_math()
    test_code_sandbox()
    print("All verifier tests passed successfully!")
