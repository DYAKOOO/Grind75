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

