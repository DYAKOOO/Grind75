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