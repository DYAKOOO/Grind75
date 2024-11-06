import sys
import os
import time
import pytest
from collections import Counter
from typing import List

# Ensure the parent directory is in the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twoPointer.valid_palindrome import Solution1, SolutionManual1
from twoPointer.twosum import Solution2, SolutionManual2

class TestTwoPointerSolutions:
    @pytest.fixture(scope="class")
    def solution1(self):
        return Solution1()
    
    @pytest.fixture(scope="class")
    def solution_manual1(self):
        return SolutionManual1()
        
    @pytest.fixture(scope="class")
    def solution2(self):
        return Solution2()
    
    @pytest.fixture(scope="class")
    def solution_manual2(self):
        return SolutionManual2()

    def print_test_header(self, test_name: str, inputs: dict):
        print(f"\n{'='*50}")
        print(f"Testing {test_name}")
        print(f"Input: {inputs}")
        print(f"{'='*50}")
        
    def debug_two_sum_result(self, numbers: List[int], target: int, result: List[int], expected: List[int]):
        print("\nDebug Information:")
        print(f"Input array: {numbers}")
        print(f"Target sum: {target}")
        print(f"Found indices: {result}")
        if result:
            print(f"Numbers at indices: {numbers[result[0]-1]} + {numbers[result[1]-1]} = {numbers[result[0]-1] + numbers[result[1]-1]}")
        print(f"Expected indices: {expected}")
        if expected:
            print(f"Expected numbers: {numbers[expected[0]-1]} + {numbers[expected[1]-1]} = {numbers[expected[0]-1] + numbers[expected[1]-1]}")
        print("-" * 50)

    @pytest.mark.parametrize("s, expected", [
        ("A man, a plan, a canal: Panama", True),
        ("race a car", False),
        (" ", True),
        (".,", True),
        ("0P", False),
        ("a.", True),
        ("", True),
        (".,", True),
        ("!@#$%^&*()", True),
        ("Madam, I'm Adam", True)
    ])
    def test_valid_palindrome(self, solution1, solution_manual1, s, expected):
        self.print_test_header("Valid Palindrome", {"s": s})
        
        start_time = time.time()
        result_optimal = solution1.isPalindrome(s)
        time_optimal = time.time() - start_time
        
        start_time = time.time()
        result_manual = solution_manual1.isPalindrome(s)
        time_manual = time.time() - start_time
        
        assert result_optimal == expected
        assert result_manual == expected
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution time: {time_manual:.6f} seconds")
        print(f"Result: {result_optimal}\n")
    
    @pytest.mark.parametrize("numbers, target, expected", [
        ([2,7,11,15], 9, [1,2]),          # 2 + 7 = 9
        ([2,3,4], 6, [1,3]),              # 2 + 4 = 6
        ([1,2,3,4], 7, [3,4]),            # 3 + 4 = 7
        ([1,3,4,6], 10, [3,4]),           # 4 + 6 = 10
        ([1,2,5,8], 7, [2,3]),            # 2 + 5 = 7
        ([2,3,6,8], 11, [2,4]),           # 3 + 8 = 11
        ([1,4,5,9], 13, [2,4]),           # 4 + 9 = 13
        ([2,4,6,9], 15, [3,4]),           # 6 + 9 = 15
        ([1,3,5,6], 8, [2,3])             # 3 + 5 = 8
    ])

 
    def test_two_sum_sorted(self, solution2, solution_manual2, numbers, target, expected):
        self.print_test_header("Two Sum II", {"numbers": numbers, "target": target})
        
        # Test optimal solution
        start_time = time.time()
        result_optimal = solution2.twoSum(numbers, target)
        time_optimal = time.time() - start_time
        
        # Print debug information before optimal solution assertion
        self.debug_two_sum_result(numbers, target, result_optimal, expected)
        
        # Test manual solution
        start_time = time.time()
        result_manual = solution_manual2.twoSum(numbers, target)
        time_manual = time.time() - start_time
        
        # Print debug information before manual solution assertion
        # self.debug_two_sum_result(numbers, target, result_manual, expected)
        
        try:
            assert result_optimal == expected
            print("Optimal solution assertion passed")
        except AssertionError:
            print(f"Optimal solution assertion failed: {result_optimal} != {expected}")
            raise
            
        try:
            assert result_manual == expected
            print("Manual solution assertion passed")
        except AssertionError:
            print(f"Manual solution assertion failed: {result_manual} != {expected}")
            raise
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution time: {time_manual:.6f} seconds")

if __name__ == "__main__":
    pytest.main(["-v", "--tb=no"])