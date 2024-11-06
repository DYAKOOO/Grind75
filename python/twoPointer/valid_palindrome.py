from typing import List

class SolutionManual1:
    """
    Manual solution for Valid Palindrom by reversing the string
    """
    def isPalindrome(self, s: str) -> bool:
        newStr = ''
        for c in s:
            if c.isalnum():
                newStr += c.lower()
        return newStr == newStr[::-1]

class Solution1:
    """
    Optimized solution for Valid Palindrome using Python's built-in methods
    Time: O(n), Space: O(1)
    """
    def isPalindrome(self, s: str) -> bool:
        l, r = 0, len(s) - 1
        
        while l < r:
            while l < r and not s[l].isalnum():
                l += 1
            while l < r and not s[r].isalnum():
                r -= 1
            if s[l].lower() != s[r].lower():
                return False
            l += 1
            r -= 1
        return True
