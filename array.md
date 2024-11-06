# Table of Contents
- .gitignore
- python/twoPointer/container_water.py
- python/twoPointer/rain_water.py
- python/twoPointer/twosum.py
- python/twoPointer/valid_palindrome.py
- python/twoPointer/threesum.py
- python/Arrays/contains_duplicate.py
- python/Arrays/longest_consecutive.py
- python/Arrays/topk_frequent.py
- python/Arrays/valid_anagram.py
- python/Arrays/group_anagram.py
- python/Arrays/two_sum.py
- python/Arrays/valid_sudoku.py
- python/Arrays/encode_decode.py
- python/Arrays/__init__.py
- python/Arrays/product_except_self.py
- python/tests/test_array.py
- c/Array/contains_duplicate.c

## File: .gitignore

- Extension: 
- Language: unknown
- Size: 38 bytes
- Created: 2024-11-04 14:35:05
- Modified: 2024-11-04 14:35:05

### Code

```unknown
__pycache__
*.pyc
.pytest_cache
notes/
```

## File: python/twoPointer/container_water.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-04 14:45:28
- Modified: 2024-11-04 14:45:28

### Code

```python

```

## File: python/twoPointer/rain_water.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-04 14:45:41
- Modified: 2024-11-04 14:45:41

### Code

```python

```

## File: python/twoPointer/twosum.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-04 14:45:00
- Modified: 2024-11-04 14:45:00

### Code

```python

```

## File: python/twoPointer/valid_palindrome.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-04 14:44:40
- Modified: 2024-11-04 14:44:40

### Code

```python

```

## File: python/twoPointer/threesum.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-04 14:45:11
- Modified: 2024-11-04 14:45:11

### Code

```python

```

## File: python/Arrays/contains_duplicate.py

- Extension: .py
- Language: python
- Size: 537 bytes
- Created: 2024-11-04 09:49:33
- Modified: 2024-11-04 09:49:33

### Code

```python
class Solution1:
    def containsDuplicate(self, nums):
        # import pdb; pdb.set_trace()  # Adding the debugger here
        hashset = set()

        for n in nums:
            if n in hashset:
                return True
            hashset.add(n)
        return False
    

# Example usage
sol = Solution1()
print(sol.containsDuplicate([1, 2, 3, 1]))  # Expected output: True
print(sol.containsDuplicate([1, 2, 3, 4]))  # Expected output: False
print(sol.containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))  # Expected output: True
```

## File: python/Arrays/longest_consecutive.py

- Extension: .py
- Language: python
- Size: 1621 bytes
- Created: 2024-11-04 14:04:17
- Modified: 2024-11-04 14:04:17

### Code

```python
# longest_consecutive.py
from typing import List

class SolutionManual9:
    """
    Naive solution using Brute search
    Time: O(n log n)
    Space: O(1)
    """
    def longestConsecutive(self, nums: List[int]) -> int:
        res = 0
        store = set(nums)

        for num in nums:
            streak, curr = 0, num
            while curr in store:
                streak += 1
                curr += 1
            res = max(res, streak)
        return res

class SolutionManual9_2:
    """
    Naive solution using sorting
    Time: O(n log n)
    Space: O(1)
    """
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
            
        nums.sort()
        longest = current = 1
        
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1]:
                continue
            elif nums[i] == nums[i-1] + 1:
                current += 1
            else:
                longest = max(longest, current)
                current = 1
                
        return max(longest, current)

class Solution9:
    """
    Optimal solution using set
    Time: O(n)
    Space: O(n)
    """
    def longestConsecutive(self, nums: List[int]) -> int:
        numSet = set(nums)
        longest = 0
        
        for n in numSet:
            # Check if it's the start of a sequence
            if (n - 1) not in numSet:
                current = 1
                # Count sequence length
                while (n + current) in numSet:
                    current += 1
                longest = max(longest, current)
        
        return longest
```

## File: python/Arrays/topk_frequent.py

- Extension: .py
- Language: python
- Size: 1205 bytes
- Created: 2024-11-04 12:46:55
- Modified: 2024-11-04 12:46:55

### Code

```python

from typing import List
import heapq
from collections import Counter, defaultdict

class SolutionManual7:
    """
    Naive solution using sorting
    Time: O(n log n)
    Space: O(n)
    """
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        # Count frequencies
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
            
        # Sort by frequency
        sorted_items = sorted(count.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k elements
        return [item[0] for item in sorted_items[:k]]

class Solution7:
    """
    Optimal solution using bucket sort
    Time: O(n)
    Space: O(n)
    """
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        freq = [[] for _ in range(len(nums) + 1)]
        
        # Group numbers by frequency
        for num, count in count.items():
            freq[count].append(num)
        
        result = []
        # Work backwards through frequencies
        for i in range(len(freq) - 1, 0, -1):
            result.extend(freq[i])
            if len(result) >= k:
                return result[:k]
        return result

```

## File: python/Arrays/valid_anagram.py

- Extension: .py
- Language: python
- Size: 520 bytes
- Created: 2024-11-04 09:47:29
- Modified: 2024-11-04 09:47:29

### Code

```python
from collections import Counter

class SolutionManual2:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return Counter(s) == Counter(t)

class Solution2:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False

        countS, countT = {}, {}

        for i in range(len(s)):
            countS[s[i]] = 1 + countS.get(s[i], 0)
            countT[t[i]] = 1 + countT.get(t[i], 0)
        return countS == countT

```

## File: python/Arrays/group_anagram.py

- Extension: .py
- Language: python
- Size: 1292 bytes
- Created: 2024-11-04 12:08:04
- Modified: 2024-11-04 12:08:04

### Code

```python
from typing import List
from collections import defaultdict

class SolutionManual4:
    """
    Naive solution using sorting for comparison
    Time: O(n * k log k) where n is length of strs and k is max length of each string
    Space: O(n * k) to store the result
    """
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        result = defaultdict(list)
        for s in strs:
            # Sort each string to use as key
            sorted_str = ''.join(sorted(s))
            result[sorted_str].append(s)
        return list(result.values())

class Solution4:
    """
    Optimized solution using character count
    Time: O(n * k) where n is length of strs and k is max length of each string
    Space: O(n * k) to store the result
    """
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        result = defaultdict(list)
        
        for s in strs:
            # Initialize count array for 26 lowercase letters
            count = [0] * 26
            
            # Count frequency of each character
            for c in s:
                count[ord(c) - ord('a')] += 1
            
            # Convert count array to tuple to use as dictionary key
            result[tuple(count)].append(s)
            
        return list(result.values())


```

## File: python/Arrays/two_sum.py

- Extension: .py
- Language: python
- Size: 1205 bytes
- Created: 2024-11-04 09:47:29
- Modified: 2024-11-04 09:47:29

### Code

```python

from typing import List

class SolutionManual3:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []  # In case no solution is found, though the problem guarantees one unique solution.


class Solution3:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        # Initialize an empty dictionary to store numbers and their indices
        num_to_index = {}
        
        # Iterate through the list of numbers with their indices
        for index, num in enumerate(nums):
            # Calculate the complement of the current number with respect to the target
            complement = target - num
            
            # Check if the complement exists in the dictionary
            if complement in num_to_index:
                # If found, return the indices of the complement and the current number
                return [num_to_index[complement], index]
            
            # If not found, add the current number and its index to the dictionary
            num_to_index[num] = index
```

## File: python/Arrays/valid_sudoku.py

- Extension: .py
- Language: python
- Size: 3382 bytes
- Created: 2024-11-04 14:09:13
- Modified: 2024-11-04 14:09:13

### Code

```python
from typing import List
from collections import defaultdict

class SolutionManual6:
    """
    Naive solution checking each constraint separately
    Time: O(n²)
    Space: O(n)
    """
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        def check_row(row: int) -> bool:
            seen = set()
            for num in board[row]:
                if num == ".":
                    continue
                if num in seen:
                    return False
                seen.add(num)
            return True
        
        def check_col(col: int) -> bool:
            seen = set()
            for row in range(9):
                num = board[row][col]
                if num == ".":
                    continue
                if num in seen:
                    return False
                seen.add(num)
            return True
        
        def check_box(start_row: int, start_col: int) -> bool:
            seen = set()
            for row in range(start_row, start_row + 3):
                for col in range(start_col, start_col + 3):
                    num = board[row][col]
                    if num == ".":
                        continue
                    if num in seen:
                        return False
                    seen.add(num)
            return True
        
        # Check all rows
        for row in range(9):
            if not check_row(row):
                return False
        
        # Check all columns
        for col in range(9):
            if not check_col(col):
                return False
        
        # Check all 3x3 boxes
        for row in (0, 3, 6):
            for col in (0, 3, 6):
                if not check_box(row, col):
                    return False
        
        return True

class SolutionManual6_2:
    """
    Optimal solution using hash sets
    Time: O(n²)
    Space: O(n²)
    """
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        cols = defaultdict(set)
        rows = defaultdict(set)
        squares = defaultdict(set)  

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                if ( board[r][c] in rows[r]
                    or board[r][c] in cols[c]
                    or board[r][c] in squares[(r // 3, c // 3)]):
                    return False

                cols[c].add(board[r][c])
                rows[r].add(board[r][c])
                squares[(r // 3, c // 3)].add(board[r][c])

        return True

class Solution6:
    """
    Optimal solution using bit masking
    Time: O(n²)
    Space: O(n)
    """
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        rows = [0] * 9
        cols = [0] * 9
        squares = [0] * 9

        for r in range(9):
            for c in range(9):
                if board[r][c] == ".":
                    continue
                
                val = int(board[r][c]) - 1
                if (1 << val) & rows[r]:
                    return False
                if (1 << val) & cols[c]:
                    return False
                if (1 << val) & squares[(r // 3) * 3 + (c // 3)]:
                    return False
                    
                rows[r] |= (1 << val)
                cols[c] |= (1 << val)
                squares[(r // 3) * 3 + (c // 3)] |= (1 << val)

        return True
```

## File: python/Arrays/encode_decode.py

- Extension: .py
- Language: python
- Size: 1566 bytes
- Created: 2024-11-04 14:09:31
- Modified: 2024-11-04 12:59:29

### Code

```python
# encode_decode.py
from typing import List

class SolutionManual8:
    """
    Encode/Decode using prefix array method
    Time: O(M) 
    Space: O(N)
    """
    def encode(self, strs: List[str]) -> str:
        if not strs:
            return ""
        sizes, res = [], ""
        for s in strs:
            sizes.append(len(s))
        for sz in sizes:
            res += str(sz)
            res += ','
        res += '#'
        for s in strs:
            res += s
        return res

    def decode(self, s: str) -> List[str]:
        if not s:
            return []
        sizes, res, i = [], [], 0
        while s[i] != '#':
            cur = ""
            while s[i] != ',':
                cur += s[i]
                i += 1
            sizes.append(int(cur))
            i += 1
        i += 1
        for sz in sizes:
            res.append(s[i:i + sz])
            i += sz
        return res

class Solution8:
    """
    Optimal solution using length encoding
    Time: O(m) for both encode() and decode()
    Space: O(1) for both encode() and decode()
    """
    def encode(self, strs: List[str]) -> str:
        res = ""
        for s in strs:
            res += str(len(s)) + "#" + s
        return res

    def decode(self, s: str) -> List[str]:
        res = []
        i = 0
        
        while i < len(s):
            j = i
            while s[j] != '#':
                j += 1
            length = int(s[i:j])
            i = j + 1
            j = i + length
            res.append(s[i:j])
            i = j
            
        return res
```

## File: python/Arrays/__init__.py

- Extension: .py
- Language: python
- Size: 0 bytes
- Created: 2024-11-04 09:47:29
- Modified: 2024-11-04 09:47:29

### Code

```python

```

## File: python/Arrays/product_except_self.py

- Extension: .py
- Language: python
- Size: 1489 bytes
- Created: 2024-11-04 13:02:01
- Modified: 2024-11-04 13:02:01

### Code

```python
# product_except_self.py
from typing import List

class SolutionManual5:
    "Pure brute force"
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n

        for i in range(n):
            prod = 1
            for j in range(n):
                if i == j:
                    continue    
                prod *= nums[j]
            
            res[i] = prod
        return res

class SolutionManual5_2:
    """
    Naive solution using division
    Time: O(n)
    Space: O(1)
    Note: This approach fails if there are zeros in the array
    """
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        total_product = 1
        for num in nums:
            total_product *= num
            
        result = []
        for num in nums:
            result.append(total_product // num)
        return result


class Solution5:
    """
    Optimal solution without using division
    Time: O(n)
    Space: O(1) excluding the output array
    """
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [1] * n
        
        # Calculate prefix products
        prefix = 1
        for i in range(n):
            result[i] = prefix
            prefix *= nums[i]
            
        # Calculate suffix products and combine
        suffix = 1
        for i in range(n-1, -1, -1):
            result[i] *= suffix
            suffix *= nums[i]
            
        return result
```

## File: python/tests/test_array.py

- Extension: .py
- Language: python
- Size: 14959 bytes
- Created: 2024-11-04 14:07:51
- Modified: 2024-11-04 14:07:51

### Code

```python
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
```

## File: c/Array/contains_duplicate.c

- Extension: .c
- Language: c
- Size: 2392 bytes
- Created: 2024-11-04 09:47:29
- Modified: 2024-11-04 09:47:29

### Code

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "uthash.h"

typedef struct {
    int key;
    UT_hash_handle hh;
} hash_table;

bool containsDuplicate(int* nums, int numsSize) {
    if (numsSize == 1) {
        return false;
    }

    hash_table *hash = NULL;
    hash_table *elem = NULL;
    bool flag = false;

    for (int i = 0; i < numsSize; i++) {
        HASH_FIND_INT(hash, &nums[i], elem);

        if (!elem) {
            elem = (hash_table *)malloc(sizeof(hash_table));
            elem->key = nums[i];
            HASH_ADD_INT(hash, key, elem);
        } else {
            flag = true;
            break;
        }
    }

    // Free up the hash table
    hash_table *tmp;
    HASH_ITER(hh, hash, elem, tmp) {
        HASH_DEL(hash, elem);
        free(elem);
    }

    return flag;
}

void test_containsDuplicate(int* nums, int numsSize, bool expected) {
    bool result = containsDuplicate(nums, numsSize);
    printf("Test: ");
    for (int i = 0; i < numsSize; i++) {
        printf("%d ", nums[i]);
    }
    printf(" -> Expected: %s, Got: %s\n", expected ? "true" : "false", result ? "true" : "false");
}

int main() {
    int test1[] = {1, 2, 3, 4};
    int test2[] = {1, 2, 3, 1};
    int test3[] = {1, 1, 1, 3, 3, 4, 3, 2, 4, 2};
    int test4[] = {1};

    test_containsDuplicate(test1, 4, false);
    test_containsDuplicate(test2, 4, true);
    test_containsDuplicate(test3, 10, true);
    test_containsDuplicate(test4, 1, false);

    return 0;
}

/*
    Time: O(n)
    Space: O(1)
*/

// typedef struct {
// 	int key;
// 	UT_hash_handle hh; // Makes this structure hashable
// } hash_table;

// hash_table *hash = NULL, *elem, *tmp;

// bool containsDuplicate(int* nums, int numsSize){
//     if (numsSize == 1) {
//       return false;
//     }
    
//     bool flag = false;
    
//     for (int i=0; i<numsSize; i++) {
//         HASH_FIND_INT(hash, &nums[i], elem);
        
//         if(!elem) {
//            elem = malloc(sizeof(hash_table));
//            elem->key = nums[i];
//            HASH_ADD_INT(hash, key, elem);
            
//            flag = false;
//        }
//        else {
//            flag = true;
//            break;
//        }
//     }
                          
//     // Free up the hash table 
// 	HASH_ITER(hh, hash, elem, tmp) {
// 		HASH_DEL(hash, elem); free(elem);
// 	}
    
//     return flag;
// }
```

