## Merge Sort
- Divide and Conquer algorithm.
- Recursively split the array into halves until each subarray has 1 element.
    - O(log(n)).
- Then merge the small sorted subarrays back together into larger sorted ones.
- The merge step compares the smallest unused elements of each half and builds a new sorted list.
    - O(n) per level.

- Time Complexity: O(nlog(n)).
- NOT INPLACE (O(n) extra space for merging).
- STABLE

- Very useful for linked lists as it does not rely on random access to elements like in heap and quick sort.
    - Efficient for pointer based DS as we do not need extra storage for merging.
    - For arrays for example, we need to develop a temporary third array to merge the left and right subpartitions.

## Quick Sort
- Recursive sorting algorithm.
- Choose some pivot element p.
    - We have a 1/2 chance of selecting a "good enough" pivot, meaning it forms decently even splits and is somewhat close to the true median.
- Partition all other items into < p and >= p.
- We now have p in the center of this partition. When we further sort, elements in the Low and High regions will never flip to the other side of p, meaning we can recursively sort the smaller partitions.
- Recursively select a new p for Low and High partitions and sort.

- Time Complexity: O(nlog(n)).
    - Worst case: O(n^2).
- INPLACE

    - In the worst case, where we repeatedly select either the smallest or larger value as our pivot, time complexity can become O(n^2).
- Since quick sort runs in O(n^2) time when the array is already sorted, we can spend O(n) time to randomise the array to greatly decrease the odds of our array being sorted. This means we (almost) no longer can face O(n^2) time and (almost) guarentee O(nlog(n)).

## QUICKSORT > MERGESORT
1. Inplace sorting.
2. Technically faster despite same Big Oh time complexity.
    - Smaller constant factors, mergesort has a lot of extra copying.
3. Can be optimised to reduce recursion depth.

## MERGESORT > QUICKSORT
1. Linked lists (cheaper merging).
2. Guarantees O(nlog(n)) worst case.
3. Stable