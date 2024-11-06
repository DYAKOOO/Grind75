
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
