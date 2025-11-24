# Quick Select
- Selection algorithm to find the k-th smallest element in an unordered list.
- Related to the quick sort algorithm.
    - Instead of recurring for both sides (after finding pivot), it recurses only for the part that contains the k-th smallest element.

```py
def partition(array, left, right):
    """
    func partition: considers last element as the pivot and moves smaller elements to the left, and greater elements to the right.
    input array: array containing values.
    input left: essentially lower bound of the array to partition.
    input right: upper bound of the array to partition.
    returns i: position of pivot element
    """

    pivot = array[right]
    i = left
    for j in range(left, right):

        if array[j] <= pivot:
            array[i], array[j] = array[j], array[i]
            i += 1

    array[i], array[right]  = array[right], array[left]
    return i

def quickSelect(array, left, right, k):
    # If k is smaller than the number of elements in the array
    if (k > 0 and k <= right - left + 1):
        index = partition(array, left, right)
        
        # If position is same as k
        if (index - left == k - 1):
            return array[index]

        # If position is more, recursively check left subarray
        if (index - left > k - 1):
            return quickSelect(array, left, index - 1, k)
        
        # Otherwise check right subarray
        return quickSelect(array, index + 1, right, k - index + left - 1)

    print("Index out of bounds")
    return
```

Time Complexity: O(n).
- Worst case O(n^2) when partitioning only seperates by one element each iteration.
Space Complexity: O(n) worst case, O(log(n)) on average.