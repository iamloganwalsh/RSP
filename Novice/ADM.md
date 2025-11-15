# Ch 1 - Introduction to Algorithm Design
## General Definitions
Verifiability: To demonstrate a particular instance is a counter example to a particular algorithm, you must be able to calculate what answer your algorithm will give in a certain instance, and display a better answer so as to prove that the algorithm didn't find it.

Simplicity: Good counter examples have all unnecessary details stripped away. Good counter examples make it exactly clear why a proposed algorithm will fail.

Modeling: Formulating an application in terms of precisely described, well-understood problems.

Permutation: Arrangements or orderings of items. {1,4,3,2} and {4,3,2,1} are two seperate permutations.
Points of interest (POI): Arrangement, tour, ordering, sequence.

Subset: Selection from a set of items.
POI: Clustering, collections, committee, group, packaging, selection.

Tree: Represents a hierarchical relationship between items.
POI: Hierarchy, dominance relationships, ancestor/descendant relationships, taxonomy.

Graph: Represents relationships between items.
POI: Network, circuit, web, relationship.

Point: Location in some geometric space.
POI: Sites, positions, data records, locations.

Polygon: Polygons define regions in some geometric space.
POI: Shapes, regions, configurations, boundaries.

String: Represents sequence of characters or patterns.
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
Left: Unsorted
Right: Sorted

Insertion sort builds the sorted list one element at a time by taking a new value and inserting it into its correct position among the already sorted elements.

For each element in the unsorted portion, swap with the new value until it fits into the correct position.

Time Complexity: O(n^2).
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
A faster growing funtion **dominates** a slower growing function.
i.e lim (n -> inf) g(n)/f(n) = 0 implies that f(n) grows much faster than g(n).

For example, n^2 grows faster than n, so n^2 dominates in this relationship. A faster growing function eventually becomes so large that the smaller function is a smyptotically irrelevant.
In Big Oh analysis, we can use this fact to simplify complexities such as O(n^2 + n) = O(n^2).
This is because as n becomes large, the n^2 term dominates the n term.

### Adding Functions
f(n) + g(n) -> Θ(max(f(n), g(n))).
Essentially, the sum of two functions is governed by the dominant function.

### Multiplying Functions
Ω(f(n)) * Ω(g(n)) -> Ω(f(n) * g(n)).
O(f(n)) * O(g(n)) -> O(f(n) * g(n)).
Θ(f(n)) * Θ(g(n)) -> Θ(f(n) * g(n)).

## Selection Sort
Repeatedly finds the smallest remaining unsorted (right side) element and puts it at the end of the sorted portion (left side) of the array.

Time Complexity: O(n^2).
Space Complexity: O(n).

## Pattern Matching
Searching for a substring in a larger text. Used in finding operations in common tools such as the browser (CRTL + f).
Time Complexity: O(nm), n = size of text, m = size of substring.
Space Complexity: O(m).

## Matrix Multiplication
Multiplying two matrices A (of dimension x * y), and B (of dimension y * z). Note that one dimension must be found in both matrices.

Time Complexity: O(n^3).
Space Complexity: O(n^2).

## Binary Search
A searching algorithm typically designed for a one-dimensional array. Iteratively cut the search space in half until either finding the target or losing the ability to cut further.

Time Complexity: O(n).
Space Complexity: O(1).

# Ch 3 - Data Structures
