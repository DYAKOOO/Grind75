import sys
import os
import time
import pytest

# Ensure the parent directory is in the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Arrays.contains_duplicate import Solution1
from Arrays.valid_anagram import Solution2, SolutionManual2
from Arrays.two_sum import SolutionManual3, Solution3

@pytest.fixture(scope="module")
def solution1():
    return Solution1()

@pytest.fixture(scope="module")
def solution2():
    return Solution2()

@pytest.fixture(scope="module")
def solution_manual2():
    return SolutionManual2()

@pytest.fixture(scope="module")
def solution_brute_force():
    return SolutionManual3()

@pytest.fixture(scope="module")
def solution_optimal():
    return Solution3()

@pytest.mark.parametrize("nums, expected", [
    ([1, 2, 3, 1], True),
    ([1, 2, 3, 4], False),
    ([1, 1, 1, 3, 3, 4, 3, 2, 4, 2], True),
    ([], False),
    ([1], False),
    ([2, 2, 2, 2], True),
])
def test_contains_duplicate(solution1, nums, expected):
    """Test the containsDuplicate method of Solution1."""
    assert solution1.containsDuplicate(nums) == expected

@pytest.mark.parametrize("s, t, expected", [
    ("anagram", "nagaram", True),
    ("rat", "car", False),
    ("", "", True),
    ("a", "a", True),
    ("a", "b", False),
    ("ab", "ba", True),
    ("listen", "silent", True),
    ("hello", "world", False),
])
def test_is_anagram(solution2, solution_manual2, s, t, expected):
    """Test the isAnagram method of Solution2 and SolutionManual2."""
    # Measure time for Solution2
    start_time = time.time()
    result_sol = solution2.isAnagram(s, t)
    end_time = time.time()
    time_sol = end_time - start_time

    # Measure time for SolutionManual2
    start_time_manual = time.time()
    result_sol_manual = solution_manual2.isAnagram(s, t)
    end_time_manual = time.time()
    time_sol_manual = end_time_manual - start_time_manual

    # Assert results
    assert result_sol == expected
    assert result_sol_manual == expected

    # Print time taken
    print(f"Time taken by Solution2: {time_sol:.6f} seconds")
    print(f"Time taken by SolutionManual2: {time_sol_manual:.6f} seconds")

@pytest.mark.parametrize("nums, target, expected", [
    ([2, 7, 11, 15], 9, [0, 1]),
    ([3, 2, 4], 6, [1, 2]),
    ([3, 3], 6, [0, 1]),
    ([1, 2, 3, 4, 5], 9, [3, 4]),
    ([10, 20, 30, 5], 50, [1, 2]),  # Corrected to [1, 3] because 20 + 30 = 50
])
def test_two_sum(solution_brute_force, solution_optimal, nums, target, expected):
    """Test the twoSum method of SolutionManual3 and Solution3."""
    # Measure time for SolutionManual3
    start_time_brute = time.time()
    result_brute = solution_brute_force.twoSum(nums, target)
    end_time_brute = time.time()
    time_brute = end_time_brute - start_time_brute

    # Measure time for Solution3
    start_time_optimal = time.time()
    result_optimal = solution_optimal.twoSum(nums, target)
    end_time_optimal = time.time()
    time_optimal = end_time_optimal - start_time_optimal

    # Assert results
    assert result_brute == expected
    assert result_optimal == expected

    # Print time taken
    print(f"Time taken by SolutionManual3: {time_brute:.6f} seconds")
    print(f"Time taken by Solution3: {time_optimal:.6f} seconds")