import sys
import os
import time
import pytest
from collections import Counter
from typing import List

# Ensure the parent directory is in the sys path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Arrays.contains_duplicate import Solution1
from Arrays.valid_anagram import Solution2, SolutionManual2
from Arrays.two_sum import SolutionManual3, Solution3
from Arrays.group_anagram import Solution4, SolutionManual4
from Arrays.product_except_self import Solution5, SolutionManual5, SolutionManual5_2
from Arrays.valid_sudoku import Solution6, SolutionManual6, SolutionManual6_2
from Arrays.topk_frequent import Solution7, SolutionManual7
from Arrays.encode_decode import Solution8, SolutionManual8
from Arrays.longest_consecutive import Solution9, SolutionManual9, SolutionManual9_2

class TestArraySolutions:
    @pytest.fixture(scope="class")
    def solution1(self):
        return Solution1()
        
    @pytest.fixture(scope="class")
    def solution2(self):
        return Solution2()
    
    @pytest.fixture(scope="class")
    def solution_manual2(self):
        return SolutionManual2()
        
    @pytest.fixture(scope="class")
    def solution3(self):
        return Solution3()
    
    @pytest.fixture(scope="class")
    def solution_manual3(self):
        return SolutionManual3()
        
    @pytest.fixture(scope="class")
    def solution4(self):
        return Solution4()
    
    @pytest.fixture(scope="class")
    def solution_manual4(self):
        return SolutionManual4()
        
    @pytest.fixture(scope="class")
    def solution5(self):
        return Solution5()
    
    @pytest.fixture(scope="class")
    def solution_manual5(self):
        return SolutionManual5()
        
    @pytest.fixture(scope="class")
    def solution_manual5_2(self):
        return SolutionManual5_2()
        
    @pytest.fixture(scope="class")
    def solution6(self):
        return Solution6()
    
    @pytest.fixture(scope="class")
    def solution_manual6(self):
        return SolutionManual6()
        
    @pytest.fixture(scope="class")
    def solution_manual6_2(self):
        return SolutionManual6_2()
        
    @pytest.fixture(scope="class")
    def solution7(self):
        return Solution7()
    
    @pytest.fixture(scope="class")
    def solution_manual7(self):
        return SolutionManual7()
        
    @pytest.fixture(scope="class")
    def solution8(self):
        return Solution8()
    
    @pytest.fixture(scope="class")
    def solution_manual8(self):
        return SolutionManual8()
        
    @pytest.fixture(scope="class")
    def solution9(self):
        return Solution9()
    
    @pytest.fixture(scope="class")
    def solution_manual9(self):
        return SolutionManual9()

    @pytest.fixture(scope="class")
    def solution_manual9_2(self):
        return SolutionManual9_2()

    def print_test_header(self, test_name: str, inputs: dict):
        print(f"\n{'='*50}")
        print(f"Testing {test_name}")
        print(f"Input: {inputs}")
        print(f"{'='*50}")

    @pytest.mark.parametrize("nums, expected", [
        ([1, 2, 3, 1], True),
        ([1, 2, 3, 4], False),
        ([1, 1, 1, 3, 3, 4, 3, 2, 4, 2], True),
        ([], False),
        ([1], False),
        ([2, 2, 2, 2], True),
    ])
    def test_contains_duplicate(self, solution1, nums, expected):
        self.print_test_header("Contains Duplicate", {"nums": nums})
        start_time = time.time()
        result = solution1.containsDuplicate(nums)
        end_time = time.time()
        assert result == expected
        print(f"Time taken: {end_time - start_time:.6f} seconds")
        print(f"Result: {result}\n")

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
    def test_is_anagram(self, solution2, solution_manual2, s, t, expected):
        self.print_test_header("Valid Anagram", {"s": s, "t": t})
        
        # Test optimal solution
        start_time = time.time()
        result_optimal = solution2.isAnagram(s, t)
        time_optimal = time.time() - start_time
        
        # Test manual solution
        start_time = time.time()
        result_manual = solution_manual2.isAnagram(s, t)
        time_manual = time.time() - start_time
        
        assert result_optimal == expected
        assert result_manual == expected
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution time: {time_manual:.6f} seconds")
        print(f"Result: {result_optimal}\n")

    @pytest.mark.parametrize("nums, target, expected", [
        ([2, 7, 11, 15], 9, [0, 1]),
        ([3, 2, 4], 6, [1, 2]),
        ([3, 3], 6, [0, 1]),
        ([1, 2, 3, 4, 5], 9, [3, 4]),
        ([10, 20, 30, 5], 50, [1, 2]),
    ])
    def test_two_sum(self, solution3, solution_manual3, nums, target, expected):
        self.print_test_header("Two Sum", {"nums": nums, "target": target})
        
        # Test optimal solution
        start_time = time.time()
        result_optimal = solution3.twoSum(nums, target)
        time_optimal = time.time() - start_time
        
        # Test manual solution
        start_time = time.time()
        result_manual = solution_manual3.twoSum(nums, target)
        time_manual = time.time() - start_time
        
        assert result_optimal == expected
        assert result_manual == expected
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution time: {time_manual:.6f} seconds")
        print(f"Result: {result_optimal}\n")

    def sort_result(self, result):
        return sorted([sorted(group) for group in result])

    @pytest.mark.parametrize("strs, expected", [
        (["eat","tea","tan","ate","nat","bat"], [["bat"],["nat","tan"],["ate","eat","tea"]]),
        ([""], [[""]]),
        (["a"], [["a"]]),
        (["ddddddddddg","dgggggggggg"], [["ddddddddddg"],["dgggggggggg"]]),
        (["hhhhu","tttti","tttit","hhhuh","hhuhh","tittt"], 
         [["tittt","tttit","tttti"],["hhhhu","hhhuh","hhuhh"]])
    ])
    def test_group_anagrams(self, solution4, solution_manual4, strs, expected):
        self.print_test_header("Group Anagrams", {"strs": strs})
        
        # Test optimal solution
        start_time = time.time()
        result_optimal = solution4.groupAnagrams(strs)
        time_optimal = time.time() - start_time
        
        # Test manual solution
        start_time = time.time()
        result_manual = solution_manual4.groupAnagrams(strs)
        time_manual = time.time() - start_time
        
        assert self.sort_result(result_optimal) == self.sort_result(expected)
        assert self.sort_result(result_manual) == self.sort_result(expected)
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution time: {time_manual:.6f} seconds")
        print(f"Result: {result_optimal}\n")

    @pytest.mark.parametrize("nums, expected", [
        ([1,2,3,4], [24,12,8,6]),
        ([2,3,4,5], [60,40,30,24]),
        ([-1,1,0,-3,3], [0,0,9,0,0]),
        ([0,0], [0,0]),
        ([1,2], [2,1])
    ])
    def test_product_except_self(self, solution5, solution_manual5, solution_manual5_2, nums, expected):
        self.print_test_header("Product Except Self", {"nums": nums})
        
        # Test optimal solution
        start_time = time.time()
        result_optimal = solution5.productExceptSelf(nums)
        time_optimal = time.time() - start_time
        
        # Test manual solutions
        try:
            # Test first manual solution
            start_time = time.time()
            result_manual = solution_manual5.productExceptSelf(nums)
            time_manual = time.time() - start_time
            assert result_manual == expected
            print(f"Manual solution 1 time: {time_manual:.6f} seconds")
            
            # Test second manual solution if no zeros
            if 0 not in nums:
                start_time = time.time()
                result_manual2 = solution_manual5_2.productExceptSelf(nums)
                time_manual2 = time.time() - start_time
                assert result_manual2 == expected
                print(f"Manual solution 2 time: {time_manual2:.6f} seconds")
        except ZeroDivisionError:
            print("Manual solution 2 skipped due to zeros in input")
            
        assert result_optimal == expected
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Result: {result_optimal}\n")

    @pytest.mark.parametrize("board, expected", [
        ([["5","3",".",".","7",".",".",".","."],
          ["6",".",".","1","9","5",".",".","."],
          [".","9","8",".",".",".",".","6","."],
          ["8",".",".",".","6",".",".",".","3"],
          ["4",".",".","8",".","3",".",".","1"],
          ["7",".",".",".","2",".",".",".","6"],
          [".","6",".",".",".",".","2","8","."],
          [".",".",".","4","1","9",".",".","5"],
          [".",".",".",".","8",".",".","7","9"]], True),
        ([["8","3",".",".","7",".",".",".","."],
          ["6",".",".","1","9","5",".",".","."],
          [".","9","8",".",".",".",".","6","."],
          ["8",".",".",".","6",".",".",".","3"],
          ["4",".",".","8",".","3",".",".","1"],
          ["7",".",".",".","2",".",".",".","6"],
          [".","6",".",".",".",".","2","8","."],
          [".",".",".","4","1","9",".",".","5"],
          [".",".",".",".","8",".",".","7","9"]], False),
        ([["." for _ in range(9)] for _ in range(9)], True)
    ])
    def test_valid_sudoku(self, solution6, solution_manual6, solution_manual6_2, board, expected):
        self.print_test_header("Valid Sudoku", {"board": "9x9 Sudoku board"})
        
        # Test all solutions
        start_time = time.time()
        result_optimal = solution6.isValidSudoku(board)
        time_optimal = time.time() - start_time
        
        start_time = time.time()
        result_manual = solution_manual6.isValidSudoku(board)
        time_manual = time.time() - start_time
        
        start_time = time.time()
        result_manual2 = solution_manual6_2.isValidSudoku(board)
        time_manual2 = time.time() - start_time
        
        assert result_optimal == expected
        assert result_manual == expected
        assert result_manual2 == expected
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution 1 time: {time_manual:.6f} seconds")
        print(f"Manual solution 2 time: {time_manual2:.6f} seconds")
        print(f"Result: {result_optimal}\n")

    @pytest.mark.parametrize("nums, k, expected", [
        ([1,1,1,2,2,3], 2, [1,2]),
        ([1], 1, [1]),
        ([1,2,2,3,3,3], 1, [3]),
        ([4,4,4,4,3,3,2,1], 2, [4,3]),
    ])
    def test_top_k_frequent(self, solution7, solution_manual7, nums, k, expected):
        self.print_test_header("Top K Frequent Elements", {"nums": nums, "k": k})
        
        def verify_result(result, nums, k):
            # Verify length
            assert len(result) == k
            # Verify frequencies
            freq = Counter(nums)
            result_freq = [freq[x] for x in result]
            expected_freq = sorted([count for num, count in freq.most_common(k)], reverse=True)
            assert sorted(result_freq, reverse=True) == expected_freq
        
        # Test both solutions
        start_time = time.time()
        result_optimal = solution7.topKFrequent(nums, k)
        time_optimal = time.time() - start_time
        
        start_time = time.time()
        result_manual = solution_manual7.topKFrequent(nums, k)
        time_manual = time.time() - start_time
        
        verify_result(result_optimal, nums, k)
        verify_result(result_manual, nums, k)
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution time: {time_manual:.6f} seconds")
        print(f"Optimal result: {result_optimal}")
        print(f"Manual result: {result_manual}\n")

    @pytest.mark.parametrize("strs", [
        ["Hello","World"],
        [""],
        ["Hello", "World!", "OpenAI", "Test", "123"],
        ["a", "b", "c", "d", "e"],
        ["Test", "with#special", "characters!"],
    ])
    def test_encode_decode(self, solution8, solution_manual8, strs):
        self.print_test_header("Encode and Decode Strings", {"strs": strs})
        
        # Test optimal solution
        start_time = time.time()
        encoded_optimal = solution8.encode(strs)
        decoded_optimal = solution8.decode(encoded_optimal)
        time_optimal = time.time() - start_time
        
        # Test manual solution
        start_time = time.time()
        encoded_manual = solution_manual8.encode(strs)
        decoded_manual = solution_manual8.decode(encoded_manual)
        time_manual = time.time() - start_time
        
        assert decoded_optimal == strs
        assert decoded_manual == strs
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution time: {time_manual:.6f} seconds")
        print(f"Original: {strs}")
        print(f"Optimal encoded: {encoded_optimal}")
        print(f"Manual encoded: {encoded_manual}\n")

    @pytest.mark.parametrize("nums, expected", [
        ([100,4,200,1,3,2], 4),
        ([0,3,7,2,5,8,4,6,0,1], 9),
        ([], 0),
        ([1], 1),
        ([1,2,3,4,5], 5),
        ([1,1,1,1,1], 1),
        ([0,-1,1,-2,2], 5),
    ])
    def test_longest_consecutive(self, solution9, solution_manual9, solution_manual9_2, nums, expected):
        self.print_test_header("Longest Consecutive Sequence", {"nums": nums})
        
        # Test optimal solution
        start_time = time.time()
        result_optimal = solution9.longestConsecutive(nums)
        time_optimal = time.time() - start_time
        
        # Test manual solutions
        start_time = time.time()
        result_manual = solution_manual9.longestConsecutive(nums)
        time_manual = time.time() - start_time
        
        start_time = time.time()
        result_manual2 = solution_manual9_2.longestConsecutive(nums)
        time_manual2 = time.time() - start_time
        
        assert result_optimal == expected
        assert result_manual == expected
        assert result_manual2 == expected
        
        print(f"Optimal solution time: {time_optimal:.6f} seconds")
        print(f"Manual solution 1 time: {time_manual:.6f} seconds")
        print(f"Manual solution 2 time: {time_manual2:.6f} seconds")
        print(f"Result: {result_optimal}\n")

def main():
    """Run the tests with proper output capture"""
    pytest.main([__file__, "-v", "-s"])

if __name__ == "__main__":
    main()