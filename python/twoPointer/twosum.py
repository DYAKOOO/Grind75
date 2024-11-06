# twosum.py
from typing import List

class SolutionManual2:
    """
    Manual solution for Two Sum II using brute force approach
    Time: O(n²), Space: O(1)
    """
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        n = len(numbers)
        for i in range(n - 1):  # Only need to go to n-1 since we need two numbers
            for j in range(i + 1, n):
                if numbers[i] + numbers[j] == target:
                    return [i + 1, j + 1]  # 1-based indexing
        return []



class Solution2:
    """
    Alternative solution using modified two pointers
    Time: O(n), Space: O(1)
    """
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        l, r = 0, len(numbers) - 1

        while l < r:
            curSum = numbers[l] + numbers[r]

            if curSum > target:
                r -= 1
            elif curSum < target:
                l += 1
            else:
                return [l + 1, r + 1]
        return []

