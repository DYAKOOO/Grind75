
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