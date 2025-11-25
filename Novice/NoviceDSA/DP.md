# Dynamic Programming - Quick Reference
- Idea is to break bigger problems into smaller overlapping subproblems
- Store solutions to avoid recomputing (using memoisation or tabulation)
    - This saves lots of time
- Either use bottom up or top down approach with caching

## Basic Steps
1. Define a subproblem
2. Find a recurrence relation
3. Identify base cases
4. Determine order of subproblems to solve
5. Compute final answer

## When to use DP
- Problem has optimal substructure (optimal solution contains optimal solutions to subproblems)
- Problem has overlapping subproblems (same subproblems are solved multiple times)
- Keywords to look out for: minimum, maximum, count ways to do xyz, longest, shortest, optimise
- Can be applied to any problem that obeys the principle of optimality
    - an optimal solution to a problem can be found by combining the optimal solutions of its subproblems

## Two main approaches
- Top-Down (Memoisation)
    - Start with original problem and recurse down
    - Cache results in a dictionary
    - E.g if (i,j) in memo: return memo[i, j]
- Bottom-Up (Tabulation)
    - Start with base cases and build up to an answer
    - Fill a table iteratively
    - Often a more space efficient approach
    - E.g dp[i][j] = xyz in nested loops

## When to avoid DP
- No overlapping subproblems (use divide and conquer)
- Need to maintain order of choices (may need backtracking)
- Whenever greedy works (simpler solution)