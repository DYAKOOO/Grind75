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