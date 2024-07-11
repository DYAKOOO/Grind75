import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Arrays.contain_duplicate import Solution

def test_contains_duplicate():
    sol = Solution()

    # Test case 1: Contains duplicate
    assert sol.containsDuplicate([1, 2, 3, 1]) == True

    # Test case 2: No duplicates
    assert sol.containsDuplicate([1, 2, 3, 4]) == False

    # Test case 3: Multiple duplicates
    assert sol.containsDuplicate([1, 1, 1, 3, 3, 4, 3, 2, 4, 2]) == True

    # Test case 4: Empty list
    assert sol.containsDuplicate([]) == False

    # Test case 5: Single element
    assert sol.containsDuplicate([1]) == False

    # Test case 6: All elements are the same
    assert sol.containsDuplicate([2, 2, 2, 2]) == True