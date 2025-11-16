# Ch 1 - Introduction to Algorithm Design
## General Definitions
Verifiability: To demonstrate a particular instance is a counter example to a particular algorithm, you must be able to calculate what answer your algorithm will give in a certain instance, and display a better answer so as to prove that the algorithm didn't find it.

Simplicity: Good counter examples have all unnecessary details stripped away. Good counter examples make it exactly clear why a proposed algorithm will fail.

Modeling: Formulating an application in terms of precisely described, well-understood problems.

Permutation: Arrangements or orderings of items. {1,4,3,2} and {4,3,2,1} are two seperate permutations.<br>
Points of interest (POI): Arrangement, tour, ordering, sequence.

Subset: Selection from a set of items.<br>
POI: Clustering, collections, committee, group, packaging, selection.

Tree: Represents a hierarchical relationship between items.<br>
POI: Hierarchy, dominance relationships, ancestor/descendant relationships, taxonomy.

Graph: Represents relationships between items.<br>
POI: Network, circuit, web, relationship.

Point: Location in some geometric space.<br>
POI: Sites, positions, data records, locations.

Polygon: Polygons define regions in some geometric space.<br>
POI: Shapes, regions, configurations, boundaries.

String: Represents sequence of characters or patterns.<br>
POI: Text, characters, patterns, labels.

Superlinear function: f(n) = nlog(n).

## Designing algorithms to solve problems
1. Modeling your application in terms of well-defined structures and algorithms is the most important single step towards finding a solution.

## Proof by Contradiction
1. Assume the hypothesis (statement you are trying to prove) is false.
2. Develop some logical consequences of this assumption.
    - Use known facts, definitions, and results to derive the consequences of assuming the hypothesis is f alse.
3. Show that this assumption leads to some contradiction, false statement, or something that clearly violates the truth.
4. If the assumption violates the truth, the hypothesis must be correct.

## Insertion Sort
Left: Unsorted<br>
Right: Sorted

Insertion sort builds the sorted list one element at a time by taking a new value and inserting it into its correct position among the already sorted elements.

For each element in the unsorted portion, swap with the new value until it fits into the correct position.

Time Complexity: O(n^2).<br>
Space Complexity: O(n).

# Ch 2 - Algorithm Analysis

## RAM (Random Access Machine) Model of Computation
- Simple operations (+, *, if, ...) take exactly one time step.
- Loops and subroutines are comprised of many single step operations.
    - Checking loop condition.
    - Computing simple operations inside each iteration.
    - etc.
- Memory access takes exactly one time step.

The RAM model assumes enough memory and doesn't discriminate between data stored in cache or on disk.

The RAM model measures run time by counting the number of time steps an algorithm takes for a specific problem instance.

This model oversimplifies its estimates as operations such as * will in most cases take more time than + operations.
When broken into assembly, multiply operations are just loops that add the number once per iteration.

Despite this, the RAM model is still an excellent model for understanding how an algorithm will work on a real computer, and is useful in practice.

## Big Oh Notation
Used for approximating runtime for an algorithm depending on input sizes in three different categories.

### Best Case Ω(n)
Best case, as the name implies, is a measure of the fastest possible runtime of an algorithm assuming the best-case input.
This can be thought of as the MINIMUM number of steps taken for the algorithm to compute an instance.

### Worst Case O(n)
Worst case is a measure of the slowest possible runtime of an algorithm assuming the worst possible input.
This can be thought of as the MAXIMUM number of steps taken for the algorithm to compute an instance.

### Average Case Θ(n)
Average case is a bit more complicated, and measures the expected runtime for an algorithm running on some general input.
This can be thought of as the AVERAGE number of steps over all possible inputs.


In most situations, the worst-case complexity is generally the most useful, as it is much easier to compute than average-case complexity and it also helps us plan around the maximum possible resource usage, ensuring the algorithm will perform acceptably even under the least favourable conditions.

Big O notation is useful for simplifying complexity calculations and provides a general idea about the performance of an algorithm.

## Dominance Relations
A faster growing funtion **dominates** a slower growing function.<br>
i.e lim (n -> inf) g(n)/f(n) = 0 implies that f(n) grows much faster than g(n).

For example, n^2 grows faster than n, so n^2 dominates in this relationship. A faster growing function eventually becomes so large that the smaller function is a smyptotically irrelevant.
In Big Oh analysis, we can use this fact to simplify complexities such as O(n^2 + n) = O(n^2).
This is because as n becomes large, the n^2 term dominates the n term.

### Adding Functions
f(n) + g(n) -> Θ(max(f(n), g(n))).<br>
Essentially, the sum of two functions is governed by the dominant function.

### Multiplying Functions
Ω(f(n)) * Ω(g(n)) -> Ω(f(n) * g(n)).<br>
O(f(n)) * O(g(n)) -> O(f(n) * g(n)).<br>
Θ(f(n)) * Θ(g(n)) -> Θ(f(n) * g(n)).

## Selection Sort
Repeatedly finds the smallest remaining unsorted (right side) element and puts it at the end of the sorted portion (left side) of the array.

Time Complexity: O(n^2).<br>
Space Complexity: O(n).

## Pattern Matching
Searching for a substring in a larger text. Used in finding operations in common tools such as the browser (CRTL + f).

Time Complexity: O(nm), n = size of text, m = size of substring.<br>
Space Complexity: O(m).

## Matrix Multiplication
Multiplying two matrices A (of dimension x * y), and B (of dimension y * z). Note that one dimension must be found in both matrices.

Time Complexity: O(n^3).<br>
Space Complexity: O(n^2).

## Binary Search
A searching algorithm typically designed for a one-dimensional array. Iteratively cut the search space in half until either finding the target or losing the ability to cut further.

Time Complexity: O(n).

# Ch 3 - Data Structures
Contiguous DS: Contiguously allocated structures are composed of single slabs of memory, including arrays, matrices, heaps, and hash tables.

Linked DS: Composed of distinct chunks of memory bound together by pointers, including lists, trees, and graph adjacency lists.

Dynamic array: Array that doubles its capacity whenever it fills up, relocates all values in memory to the new memory address, and returns the previous memory space to the storage allocation system.

Container: Abstract data type that permits storage and retrieval of data items independent of content. Distinguished by retrieval order (e.g LIFO, FIFO).

## Binary Search Tree
**Search** Time Complexity: O(h), height of the tree.<br>
**Insert** Time Complexity: O(h).<br>
**Delete** Time Complexity: O(h).
- Note: When deleting a node with two children, we find the next highest value (the next value InOrder) to replace the deleted node.
- Alternative BST's such as red-black and splay trees ensure BST is balanced, allowing for log(n) operations instead of O(h).

## Hashing
**Compaction/Fingerprinting**: Represent large objects using small hash codes. It is easier to work with small objects rather than large one, and the hash code typically preserves the identity of an item.

### Hashing - Collision Resolution
**Chaining**: Hash table represented as an array of linked lists. If a collission occurs, and the new item to the chain.

**Open Addressing**: Maintains hash table as simple array, initialising all values to NULL. If a collision occurs, we use sequential probing to insert the item into the next open slot in the table. On Deletion, we need to reinsert all items on the "run" to avoid gaps which could disrupt our future searching or deleting attempts.
- Run refers to the sequential values following the hash index with no gaps inbetween.

## Specialised DS
**Geometric**: Collections of data points and regions.<br>
**Graph**: Typically implemented with adjacency matrices or lists.<br>
**Set**: Subsets of items typically represented using a dictionary to support fast membership queries.

