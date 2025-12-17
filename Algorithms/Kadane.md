# Kadane's Algorithm
- Used to find the maximum sum of a contiguous subarray with a one-dimensional array of numbers.

## General Approach:
- Maintain two variables during a single pass:
    - curr_max: Stores maximum sum of a subarray ending at the current index.
        - At each element, curr_max is updated to be the maximum of either the current element itself, or the current element added to the curr_max from the previous position.
        - If curr_max becomes negative, it is reset to 0, indicating that a new subarray would produce a better result.
    - global_max: This stores the overall max sum found so far across the entire array, updating global_max = max(global_max, curr_max) for each curr_max

## Algorithm Steps:
1. Initialise global_max to a very small number or the first element
2. Initialise curr_max to 0
3. Iterate through each element in the array
    - Add current element to curr_max
    - If curr_max > global_max: update global_max
    - If curr_max < 0: curr_max = 0