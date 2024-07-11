class Solution:
    def containsDuplicate(self, nums):
        # import pdb; pdb.set_trace()  # Adding the debugger here
        hashset = set()

        for n in nums:
            if n in hashset:
                return True
            hashset.add(n)
        return False

# Example usage
sol = Solution()
print(sol.containsDuplicate([1, 2, 3, 1]))  # Expected output: True
print(sol.containsDuplicate([1, 2, 3, 4]))  # Expected output: False
print(sol.containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]))  # Expected output: True