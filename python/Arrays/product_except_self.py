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