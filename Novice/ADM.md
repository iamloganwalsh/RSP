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

# Ch 4 - Sorting
**Stable**: A sorting algorithm is stable if sorted elements retain their relative order. For example, if we have two duplicates with the same value, call them A and B. Assume A occurs before B in the unsorted dataset. Then when we sort the dataset, A will still come before B.

## Heap Sort
- Min or Max Heaps.
- Works like a priority queue.
- Heaps aren't necessarily sorted but lie somewhere between sorted and random.
    - In Min heap: each node is larger than it's parents but smaller than it's children, though is not sorted between it's siblings.
- Heaps are typically binary trees
- Always fills left to right
- Accessing parent (from array): child_index // 2
- Access children (from array): parent_index * 2
- Involves constructing a heap and then popping the root repeatedly, adding the value to the sorted array.
    - Time Complexity: O(nlog(n))

### Heap Init & Insert
- Sequentially add each value to the heap, percolating on each iteration to ensure integrity.

### Heap Delete
- When we pop the root (i.e the max or min of the heap), we replace with the right-most (last) child.
- Iteratively percolate down to ensure heap integrity.

### Faster Heap Construction (Bottom-Up Build)
- Instead of inserting one element at at ime, treat the array as a heap and fix it from the bottom up.
- Start from the last non-leaf node and percolate down.
- This turns the arbitrary array into a valid heap in O(n) time.
- Construction time does not dominate complexity of heap sort, so sorting still takes O(nlog(n)).

## Merge Sort
- Divide and Conquer algorithm.
- Recursively split the array into halves until each subarray has 1 element.
    - O(log(n)).
- Then merge the small sorted subarrays back together into larger sorted ones.
- The merge step compares the smallest unused elements of each half and builds a new sorted list.
    - O(n) per level.
- Time Complexity: O(nlog(n)).
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
    - In the worst case, where we repeatedly select either the smallest or larger value as our pivot, time complexity can become O(n^2).
- Since quick sort runs in O(n^2) time when the array is already sorted, we can spend O(n) time to randomise the array to greatly decrease the odds of our array being sorted. This means we (almost) no longer can face O(n^2) time and (almost) guarentee O(nlog(n)).

## Randomisation
**Random Sampling**: Select a small random sample and find the median, assume it reprents the full set.<br>
**Randomised Hashing**: In the worst case, all of our hashed values face collisions. If we randomly select a hash function out of a set of good hash functions, we reduce the odds of facing many collisions.<br>
**Randomised Searching**: Simulated annealing.

## Bucket Sort
Example: Sorting names in a phonebook.
- First, partition them according to first letter of the larst name.
- This creates 26 buckets of names.
    - Any name in the C pile must occur after any name in the B pile.
    - Therefore we can sort individual buckets and concatenate them all together to form a sorted list.
- Assuming names are evenly distributed, 26 smaller sorting problems are much better than sorting an entire array.
- We can further partition each pile based on the second letter of each name, getting smaller and smaller piles.
- The set of names will be completely sorted once every bucket contains only a single name.

Bucket sort is very effective when we expect the distribution of data to be uniform.

# Ch 5 - Divide and Conquer
- Divide a problem into 2 smaller subproblems, solve each recursively, then combine the partitions into the final solution.
- Algorithm is considered efficient if merging takes less time than solving the 2 subproblems.

## Binary Search

### Counting Occurances - BS
- Can binary search to find an element, then sequentially check left and right.
- This can take up to O(n) if the array contains only a single unique element.
- We can modify binary search to find the right boundary of the block, then iterate left until you reach the left boundary.
    - Delete the arr[middle] == key check. This means that our search always fails and returns -1, so instead we want to return (l + r) / 2.
    - Whenever we recalculate to another index with the same value, it will proceed to the right.
    - Our algorithm will eventually terminate once it reaches the right boundary.

### One-Sided BS
- Suppose we have an array containing a series of 0's followed by a series of 1's, and we want to find the point where it transitions.
- If bounded, we could find this point in log(n) tests.
- Without a bound, we repeatedly test larger intervals until we find a non-zero value.
    - By doubling our interval each time, we can formulate a window to perform binary search on.
    - For example, if A[8] = 0 and A[16] = 1, this can become our window.
    - This means we can find the transition point in at most 2log(p) comparisons.

### Finding Square Roots
- We can find the square root of a number using binary search.
- Simply set the lower boundary to 1 and upper boundary to the number whose root we are trying to find.
- From here, we can use binary search to find the root.

## Fast Multiplication
1. Split numbers in half, breaking them into smaller pieces.
2. Multiply the smaller pieces.
3. Apply Karatsuba's trick.
    - Normally you'd need 4 small multiplications, but this trick allows us to use only 3.
4. The general idea is that we can perform fewer than needed multiplications, allowing us to speed up the process for larger numbers.

## Largest Subrange and Closest Pair
- Task: take an array A of n numbers, and find the index pair that maximises array[i:j] (inclusive).
- Brute force solutions gives O(n^2).
- Divide and Conquer solution:
    - Max subrange is either entirely in the left or right halves or crosses the middle.
    - If we have to cross the middle, we need to find the best from the left side which meets the middle and do the same for the right side.
    - We find this crossover in O(n).
    - Therefore our recurrence gives us O(nlog(n))

## Parallel Algorithms
- Parallel Computation is  the process of breaking large problems into smaller subtasks, using multiple processors to solve it faster.
- Divide and Conquer is most suited to parallel computation.
- We want to partition our problem of size n into p equally sized parts, and feed one to each processor.
- This reduces time cost from T(n) to T(n/p), plus the cost of combining results together.

**Data Parallelism**: Running a single algorithm on different and independent datasets.

## Convolution
- A way of combining two functions or sequences to form a third function.
- C[n] = Sum_k(A[k] * B[n-k])
- Imagine sliding one function over another, multiplying the overlapping parts, and adding them up. The amount of overlap at each position becomes the output.
- Used in signal & image processing, machine learning (CNNs), and more.

# Ch 6 - Hashing and Randomised Algorithms
- Most previously discussed algorithms are designed to optimise worst case performance.
- Relaxing this demand can lead to useful algorithms that still have performance guarantees.
- Randomised algorithms aren't heuristics, and bad performance can be due to being unlucky rather than because of input data.
- There are two types of randomised algorithms, differing by their guarentee on correctness or efficiency.
    1. Las Vegas algorithms: guarentee correctness, and are usually (but not always) efficient.
    2. Monte Carlo algorithms: probably efficient, and usually (but not always) produce the correct answer or something close to it.
- Eliminating the need to worry about rare or unlikely situations makes it possible to avoid complicated data structures.
- Randomised algorithms are difficult to analyse rigorously.

## Probability - go over again

## Balls and Bins
- Say we have x balls and y bins. We are interested in the distribution of balls in bins assuming we throw all x balls towards y bins.
- Hashing can be thought of as this Ball and Bin process. Suppose we have n balls and n bins.
    - We expect on average 1 ball per bin, but this is true regardless of hashing function quality.
    - A good hash function should behave as a RNG, selecting integers with equal probability from 1 to n.
- 36.78% of bins in large hash tables are expected to be empty or have only one element.

# Ch 7 - Graph Traversal
- A graph G = (V, E), where V is a set of vertices and E is a set of  vertex pairs, also known as edges.
- Graphs can be used to represent essentially any relationship.

## Types of Graphs
**Weighted**: Each edge or vertex in a weighted graph is assigned a numerical value, also known as a weight. This is important for shortest path problems. For unweighted graphs, the shortest path is calculated based on the shortest number of edges, and can be solved using BFS only. Weighted graphs on the other hand require more complicated algorithms such as dijkstras.

**Simple vs Non-Simple**: A graph is simple if it doesn't contain self-loops (edges on a node that point back to itself, an edge (x,x)) or multiedges (multiple bridges between two nodes, e.g (x,y) exists more than once in our set of edges).

**Sparse vs Dense**: Graphs are sparse only when a small fraction of possible vertex pairs have edges. Dense graphs typically have n^2 edges, while sparse graphs are linear in size.

**Complete**: A graph is complete if it contains all possible edges.

**Connected**: A graph is connected if every pair of vertices is connected by some path.

**Cyclic vs Acyclic**: A cycle is a closed path of 3 or more vertices that has no repeating vertices except for the start/end point. Trees are undirected graphs which are connected and acyclic.  Directed acyclic graphs (DAGs) arise in scheduling problems, where a directed edge (x,y) indicates that an activity x must occur before y.

**Topological**: The edge-vertex representation G = (V, E) describes the purely topological aspects of a graph.

**Embedded**: A graph is embedded if the vertices and edges are assigned geometric positions.

**Implicit vs Explicit**: Certain graphs are not explicitly constructed then traversed, but built as we use them.

**Labelled vs Unlabelled**: Each vertex is assigned a unique name/identifier in a labelled graph to distinguish it from all other vertices.

**Friendship**: A friendship graph models people as vertices and friendships as edges. 

## Graph Data Structures
Assume some graph G = (V, E) contains n vertices and m edges.
1. Adjacency List: represent G using a list of lists L, where L[i] is a list of all vertices j such that (i,j) is an edge of G.
2. Adjacency Matrix: represent G using an n x n matrix M, where element M[i,j] = 1 if (i,j) is an edge of G, and 0 if it isn't.

### Adjacency List Advantages:
- Faster to find the degree of a vertex.
- Less memory on sparse graphs.
- Faster traversal.
- Better for most problems.

### Adjacency Matrix Advantages:
- Faster to test if an edge is in a graph.
- Less memory on dense graphs.
- Edge insertion/deletion.

It is harder to check if an edge exists in adjacency lists, but it is easy to design algorithms that avoid such queries.

## Graph Traversal
- Mark each vertex when we first visit it, and keep track of what vertices we haven't explored.
- Each vertex has one of three states:
    1. Undiscovered.
    2. Discovered: found but not explored.
    3. Processed: we have discovered all of the vertex's incident edges.

### BFS
- Maintain a queue and visit nodes in order of the queue.
- Only useful in unweighted graphs.
- Applications:
    - If properly implemented using adjacency lists, any graph traversal algorithm is linear since BFS is O(n + m) for both directed and undirected edges.
    - Connected Components: BFS can find all connected vertices as it traverses all discovered edges.
    - Two-Colouring Graphs: Assigns a colour to each vertex of a graph such that no edge links any two vertices of the same colour.
        - A graph is bipartite if it can be coloured without conflicts using only two colours.
        - Can be achieved by augmenting BFS to assign the opposite colour to a nodes parent.

### DFS
- Expands nodes as they appear then go back and try siblings.
- Back edge: edge whose other endpoint is an ancestor of the vertex being expanded.
- Applications:
    - Finding cycles in graphs. Undirected: finding vertex is visited but not the parent of the current vertex. Directed: finding a vertex that is visited which is already in the recursion stack.
    - Articulation (Cut-node) Vertices: A single vertex whose deletion disconnects a connected component of the graph. **Go over this again**
        - Connectivity: smallest number of vertices whose deletion will disconnect the graph.
        - For each vertex u, we compute disc[u] (discovery time when first visiting u) and low[u] (the earliest discovered vertex reachable from u). low[u] detects whether a child subtree can "escape" upward via a back edge.
        - For root of the tree, if there is two or more children then it must be an articulation vertex.
        - For a non-root vertex, it is an articulation point if low[u] > disc[u] where v is one of u's children. This means the child subtree rooted at v cannot reach any ancestor of u with a back edge. If u is removed, v's entire subtree becomes disconnected, meaning u is a cut vertex.

# Ch 8 - Weighted Graph Algorithms

## Minimum Spanning Trees
- A spanning tree of a connected graph G = (V, E) is a subset of edges from E forming a tree connecting all vertices of V.
- The minimum spanning tree has the smallest sum of edge weights.
- MSTs are used whenever we need to connect a set of points cheaply using the smallest amount of roadway, wire, or pipe.
- Any tree is the smallest in terms of number of edges, but the MST is the smallest connected graph in terms of edge weight.
- Graphs can have multiple MSTs.
- For undirected graphs, all spanning trees are MST.
- For directed graphs, we can find MST using 2 algorithms:
    1. Prims
        - Pros: Faster for dense graphs, easier to implement on adjacency matrices.
        - Cons: Slower for sparse graphs, requires connected graph.
    2. Kruskal's
        - Pros: Faster for sparse graphs, works on disconnected graphs.
        - Cons: Slower for dense graphs.

### Prim's Algorithm
- Start from one vertex and grow the rest of the tree one edge at a time until all verticles are included.
- Repeatedly select the smallest weight edge that will enlarge the number of vertices in the tree.
- Time Complexity: O(E + Vlog(V)) with priority queue.
    - Adding edges to the PQ takes O(Elog(V)).
    - Extracting minimum edge takes O(log(V)), and we do it V times for a total of O(Vlog(V)).
```py
import heapq
"""
Find the MST of a graph using Prim's algorithm.
adj_list: Dictionary representing the adjacency list of the graph.
    Keys are node labels, and values are lists of (neighbour, weight) tuples.
Returns list of edges in MST and total weight of the MST.
"""

def prim_mst(adj_list):
    if not adj_list:
        return [], 0

    mst_edges = []
    total_weight = 0
    visited = set()

    start_node = next(iter(adj_list))       # Start with first node in adj_list
    min_heap = [(0, start_node, None)]      # (weight, current_node, parent)

    while min_heap:
        weight, current_node, parent = heapq.heappop(min_heap)

        if current_node in visited:
            continue

        visited.add(current_node)

        Add valid edges to the MST
        if parent is not None:
            mst_edges.append((parent, current_node, weight))
            total_weight += weight

        # Add adjacent edges
        for neighbour, edge_weight in adj_list[current_node]:
            if neighbour not in visited:
                heapq.heappush(min_heap, (edge_weight, neighbour, current_node))
        
    if len(visited) != len(adj_list):
        raise ValueError("Graph is not connected, cannot form MST!")
    
    return mst_edges, total_weight
```

### Kruskal's Algorithm
- More efficient on sparse graphs.
- Doesn't start on a particular vertex.
- Works with disconnected graphs.
- May produce different MST than Prim's, but both will have the same weight.
- Kruskal's algorithm builds up connected components of vertices, resulting in the complete MST.
    - Initially, each vertex forms its own seperate component in the tree.
    The algorithm repeatedly considers the lightest remaining edge, and tests if its two endpoints are already connected within the same connected component.
        - If yes, the edge is discarded because adding it would create a cycle.
        - Otherwise, insert the edge and merge the two components into one.
- Time Complexity: O(Elog(E)) using union-find.
    - NOTE: Union find selects a collection of disjoint sets (no common elements) and merges them together.
    - Sorting edges takes O(Elog(E)).
    - The while loop runs at most E times.
```py
def kruskal_mst(edges, num_vertices):
    """
    Finds the MST of a graph using Kruskal's algorithm.
    edges: List of edges, where each edge is represented as (weight, u, v).
        u and v are vertices connected by the edge, and weight represents the edge weight.
    num_vertices: Total number of vertices in the graph.
    Returns a list of edges in the MST and the total weight.
    """

    # Sort edges by weight (first element)
    edges.sort()

    parent = list(range(num_vertices))      # Parent array for Union-Find
    rank = [0] * num_vertices               # Rank the array for Union-Find

    def find(node):
        # Finds the root of the set containing the node
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        # Union the sets containing two nodes
        root1 = find(node1)
        root2 = find(node2)

        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            elif rank[root1] < rank[root2]:
                parent[root1] = root2
            else:
                parent[root2] = root1
                rank[root1] += 1

    # Kruskal's algorithm
    mst_edges = []
    total_weight = 0

    for weight, u, v in edges:      # Repeatedly select lowest weight edges to try connect the graph
        if find(u) != find(v):      # No common parent
            union(u, v)
            mst_edges.append((u, v, weight))
            total_weight += weight

        # We can stop early if we find enough edges
        if len(mst_edges) == num_vertices - 1:
            break

    return mst_edges, total_weight
```

## MST Variations

### Maximum Spanning Tree
- Can be found by negating the weights of all edges, and running Prim's or Kruskal's.

### Minimum Product Spanning Trees
- A spanning tree with a weight product that is <= the weight product of all otgether spanning trees.
- Suppose we seek the spanning tree that minimises the product of edge weights, assuming all edge weights are positive.
- Since log(ab) = log(a) + log(b), the MST on a graph whose edge weights are replaced with logarithms gives the minimum product spanning tree on the original graph.
    - We use log to convert our product into a sum since Prim's and Kruskal's require sum weights.

### Minimum Bottleneck Spanning Tree
- A spanning tree that minimises the maximum edge weight over all possible trees.
    - A spanning tree where the most expensive edge is as cheap as possible.
- Every MST has this property.

## Union-Find
- A set partition seperates the elements of some universal set into a collection of disjoint subsets, where each element is in exactly one subset.
- Connected components in a graph can be represented as a set partition.
- For Kruskal's algorithm to run efficiently, we need a data structure that supports the following operations:
    - Same component(v1, v2) - do vertices v1 and v2 occur in the same connected component of the current graph?
    - Merge components(C1, C2) - merge the given pair of connected components into one in reponse to the insertion of an edge between them.
- The Union-Find DS represents each subset as a backawrds tree, with pointers from a node to its parent.
    - Each node of this tree contains a set element, and the name of the set is taken from the key at the root.
    - Also, track the number of elements in the subtree rooted at each vertex v.
    - The two operations are implemented as:
        1. find(i): find the root of the tree containing element i, by walking up the parent pointers until there is nowhere to go. Return the label of the root.
        2. union(i): link the root of one of the trees (e.g containing i) to the root of the tree containing the other (say j), so find(i) now equals find(j).
        - We want to minimise the tree height to prevent unbalanced trees.
            - To do this, make the smaller tree the subtree of the bigger one.

```py
class UnionFind:
    def __init__(self, n):
        """
        Initialise Union-Find for n elements.
        Each element starts as its own parent, and the rank is initialised to 0.
        """
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        """
        Find the root of the set containing x with path compression.
        """
        if self.parent[x] != x:                             # If x has a known parent
            self.parent[x] = self.find(self.parent[x])      # Recursively find their parent
        return self.parent[x]                               # Return their parent

    def union(self, x, y):
        """
        Union the sets containing x and y using union by rank.
        """

        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:                        # Perform union on disjointed sets
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x        # If we face a tie, just choose whichever
                self.rank[root_x] += 1
        
    def connected(self, x, y):
        """
        Check if x and y is connected
        """
        return self.find(x) == self.find(y)
```

## Shortest Paths
- A path is a sequence of edges connecting two vertices.
- The shortest path is the one which minimises the sum of edge weights.
- In an undirected graph, the shortest path from s to t can be found using a BFS starting from s.

### Dijkstra's Algorithm
- Finds the shortest path in an edge or vertex weighted graph.
- Starting from a particular vertex s, it finds the shortest path from s to all other vertices in the graph, including the destination t.
- Consists of a series of rounds, where each round establishes the shortest path from s to some new vertex.
- Time Complexity: O(E + Vlog(V)).
    - There are V nodes, and each one needs to be processed (extracted from the priority queue) in log(V) time, so this step takes O(Vlog(V)).
    - For each node, the algorithm checks its neighbours and updates their distances. Each edge is checked once during the algorithm, which takes O(E).
```py
import heapq

def dijkstras(graph, start):
    """
    Finds shortest path from start to all other nodes in a graph.
    graph: dictionary adjacency list. Keys are nodes and Values are lists of tuples (neighbour, weight).
    start: Starting node.
    Returns a dictionary where keys are nodes and values are the shortest paths from start -> node.
    """

    distances = {node: float('infinity') for node in graph}     # Init all distances to infinite
    distances[start] = 0
    pq = [(0, start)]
    
    while pq:
        curr_dist, curr_node = heapq.heappop(pq)                # Heappop to find lowest (pq)

        if curr_dist > distances[curr_node]:                    # Found a shorter path previously
            continue

        for neighbour, weight in graph.get(curr_node, []):      # Only check valid nodes
            dist = curr_dist + weight

            if dist < distances[neighbour]:                     # Enqueue new nodes if its shorter than    the known path
                distances[neighbour] = dist
                heapq.heappush(pq, (dist, neighbour))
    
    return distances
```

### Floyd-Warshall Algorithm
- Works on graphs with negative weights.
- Can be used to find:
    - Graph's diameter: The largest shortest-path distance over **all pairs** of vertices.
    - Centre vertex: The vertex that minimises the longest or average distance to all the other nodes.
- This involves calculating the shortest path between all pairs of vertices in a graph.
- Best implemented using an adjacency matrix since we need to store n^2 distances anyways.
- In the martix, initialise every edge with its weight, and every non-edge with infinity.
    - The initial all pairs shorest path matrix will be the initial adjacency matrix.
    - We will perform n iterations, where the kth iteration only allows for the first k vertices as possible intermediate steps on the path between each pair of vertices x and y.
- Time Complexity: O(V^3).
    - There are V iterations for k, and V^2 updates for each (i, j) pair per iteration.
```py
import math

def floyd_warshall(graph):
    """
    Implements the Floyd-Warshall algorithm to find all-pairs shortest paths.
    graph: 2D list (matrix) representing the adjacency martix of the graph.
        graph[i][j] is the weight of the edge from i to j.
        math.inf represents no direct edge between i and j.
        Self-loops (graph[i][i]) should be 0.
    Returns a 2d list representing shortest distances between all pairs of vertices.
        Returns None if a negative cycle is detected.
    """

    num_vertices = len(graph)
    dist = [row[:] for row in graph]        # Create a copy of the graph for tracking distances

    for k in range(num_vertices):           # Iterate immediate vertices
        for i in range(num_vertices):       # Iterate possible source vertices
            for j in range(num_vertices):   # Iterate all possible destination vertices

                # If vertex k is on the shortest path from i to j, update dist[i][j]
                if dist[i][k] != math.inf and dist[k][j] != math.inf:
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    # Detect negative cycles
    for i in range(num_vertices):
        if dist[i][i] < 0:
            return None

    return dist
```

### Bellman-Ford Algorithm
- Single source shortest path algorithm.
- Given a weighted graph G = (V, E) and a source vertex start, the task is to compute shortest distance from the **starting vertex** to all other vertices.
    - If negative cycle occurs, return None
    - If a vertex is unreachable, the distance should be infinite
- Time Complexity: O(V * E).
- Principle of Relaxation of Edges: Relaxation means updating the shortest distance to a node if a shorter path is found through another node.
```py
import math

def bellman_ford(num_vertices, edges, start):

    # Initialise distance
    dist = [math.inf] * num_vertices
    dist[start] = 0

    for i in range(num_vertices):
        for edge in edges:
            u, v, weight = edge

            # Relax edges (replace with shorter path)
            if dist[u] != math.inf and dist[u] + weight < dist[v]:

                if i == num_vertices - 1:           # If our path can still be improved, we have a negative cycle
                    return [None]

                dist[v] = dist[u] + weight
    
    return dist
```

## Network Flows and Bipartite Matching
- An edge weighted graph can be interpreted as a network of pipes, where the weights of an edge determines the capacity of the pipe.
- The network flow problem asks for the maximum amount of flow that can be sent from vertices s to t in a given weighted graph G while respecting the maximum capacities of each pipe.

### Bipartite Matching
- A matching in a graph G = (V, E) is a subset of edges E' \subset E such taht no two edges E' share a vertex.
    - A set of edges that don't share any vertices.
- G is bipartite or two-colourable if the verices can be divided into two sets L and R such that all edges in G have one vertex in L and one vertex in R.
- Finding the maximum cardinality bipartite matching using network flow:
    - Create source node s that is connected to every vertex in L with an edge weight of 1.
    - Create sink node t that is connected to every vertex in R with an edge weight of 1.
    - The maximum possible flow from s to t defines the largest matching in G.

### Computing Network Flows
- Augmenting path: A path in a flow network that can carry more flow, increasing the total flow from source to sink.
- A flow through a network is optimal iff it contains no augmenting path.
- Each augmentation increases flow, so we repeat until no such path remains to find the global maximum.
- Residual flow graph:
    - Flow network constructed from graph G and flow f.
    - For each edge e of G:
        - If c(e) - f(e) > 0, create forward edge e with capacity equal to c(e) - f(e), i.e remaining capacity.
        - If f(e) > 0, create backwards edge with capacity equal to f(e), i.e capacity used.
    - A path in the residual flow graph from s to t implies that more flow can be pushed from s to t.
    - The smallest edge weight on this path defines the amount of extra flow that can be pushed along it.

# Ch 9 - Combinatorial Search

## Backtracking
- Systematically running through all possible configurations of a search space.
- These may represent:
    - All possible arrangements of objects (permutations).
    - All possible ways of building a collection of objects (subsets).
- Other common situations to use backtracking:
    - Enumerating all spanning trees of a graph.
    - Enumerating all paths between two vertices.
    - All possible ways to partition vertices into colour classes.
- We generate each possible configuration exactly once, avoiding repititions and missed configurations.
    - We need a systematic generation order.
- Combinatioral search solutions are modelled as a vector a = (a1, a2, ..., a_n) where each element a_i is selected from a finite ordered set S_i.
    - This vector might represent an arrangement where a_i contains the ith element of the permutation.
    - Or a is a Boolean vector representing a given subset S, where a_i is true iff the ith element of the universal set is in S.
    - It can also represent a seuqnece of moves in a game, or a path in a game where a_i contains the ith game move or graph edge in the sequence.
- At each step in the backtracking algorithm, we try to extend a given partial solution by adding another element at the end.
    - Then, we test whether we now have a complete solution:
        - If so, we print or return or whatever the solution.
        - Else we must check if the partial solution is potentially extendable to form a complete solution.
            - Using methods such as forward checking can identify failed solutions early on.
- Backtracking constructs a tree of partial solutions, where each node represents a partial solution.
    - There is an edge from x to y if node y was created by extending x.
    - This tree of partial solutions provides an alternative way to think about backtracking.
        - The process of construction solutions corresponds to doing a DFS traversal of the backtrack tree.
            - BFS can be used but uses more space.
        - This yields a recursive implementation of a basic backtracking algorithm.
    - The current state of the search is represented by a path from root to current DFS node.
```py
def backtrack(a, k):
    """
    Generic backtracking implementation.
    a: list representing a partial solution.
    k: integer representing current position in the solution a.
    """

    if is_solution(a, k):                       # If the partial solution is actually a full solution,
        process_solution(a)                     # we either return or print or somehow handle the solution
        return

    for choice in generate_choices(a, k):       # All possible values for a[k]
        if is_valid(a, k, choice):              # Is this a valid partial solution
            a[k] = choice                       # Make the choice (extend the partial solution)
            backtrack(a, k + 1)                 # Go deeper
            a[k] = None                         # Undo the choice (backtrack)
```
**Functions in above code**:
- is_solution(a,k): tests whether the first k elements of vector a form a complete solution for the given problem.
- process_solution(a): handles the case where our partial solution qualifies as a complete solution. This could be printing, counting, storing, or somehow processing the complete solution.
- generate_choices(a, k): returns some array filled with possible candidates for the kth position of partial solution a given the previous k - 1 elements.

### BT Example 1 - Constructing all Subsets
- How many subsets are there of an n-element set, say integers {1, ..., n}.
    - Two subsets for n = 1, being {} and {1}, there are 4 subsets for n = 2, 8 subsets for n = 3.
        - 2^n subsets of n elements.
    - Each subset described by elements inside it.
    - To construct all subsets, we set up a Boolean array of n cells, wwhere a value a_i signifies whether the ith item is in the given subset.
    - For the general algorithm, S_k = (true, false) and a is a solution whenever k = n.
```py
def gen_subsets(n):
    a = [False] * n
    results = []

    def backtrack(k):
        # If we have assigned all n positions, record the subset
        if k == n:
            subset = [i + 1 for i in range(n) if a[i]]
            results.append(subset)
            return
        
        # Case 1: Include k+1
        a[k] = True
        backtrack(k+1)

        # Case 2: Disclude k+1
        a[k] = False
        backtrack(k+1)

    backtrack(0)
    return results
```

### BT Example 2 - Constructing all Permutations
- Counting permutations of {1, ..., n} is necessary to generate them.
- There are n distinct choices for the value of the first element of a permutation, then n-1 candidates for the second position, so there are n! distinct permutations.
- Set up an array a of n cells.
    - The set of candidates for the ith position will be all elements that have not appeared in the i - 1 elements of the partial solution, corresponding to the first i - 1 elements of the permutation.
- For the genberal algorithm, S_k = {1, ..., n} - {a_1, ..., a_k} and a is a solution whenever k = n.
```py
def permutations(n):
    a = [None] * n                  # Holds current permutation
    used = [False] * (n + 1)        # Checks taken numbers

    def backtrack(k):
        if k == n:                  # Base case: Wer have used all the numbers
            print(a)                # aka permutation is complpete
            return

        for x in range(1, n+1):     # Iterate every candidate, if statement filters for unused
            if not used[x]:
                a[k] = x
                used[x] = True

                backtrack(k+1)      # Backtrack with current value of x
                used[x] = False     # Reset used[x] for the next iteration (undo choice)

    backtrack(0)
```
### BT Example 3 - Constructing all Paths in a Graph
- In a simple path, no vertex appears more than once.
- Enumerating all simple s to t paths in a given graph is more complicated than permutations or subsets.
- The input data we must pass to backtrack to construct the paths consists of the input graph g, source vertex s, and target vertex t.
- The starting point of any path from s to t is always s, thus s is the only candidate for the first position.
    - The possible candidates for the second position are the vertices such that (s,v) is an edge for a graph.
    - Whenever a_k = t, we've found a successful path.
- In general, S_k+1 consists of the set of vertices adjacent to a_k that have not been used elsewhere in the partial solution a.
```py
def all_paths(g, s, t):
    path = [s]
    visited = set([s])

    def backtrack(vertex):
        if vertex == t:
            print(path)
            return
        
        for neighbour in g[vertex]:             # Iterate all edges
            if neighbour not in visited:
                visited.add(neighbour)          # Add neighbour to visited set
                path.append(neighbour)          # Add the neighbour to the path for next iteration
                backtrack(neighbour)
                visited.remove(neighbour)       # Undo the visit
                path.pop()

    backtrack(s)
```

## Search Pruning
- It is inefficient to construct all permutations first to be analysed later.
- Suppose our search started from vertex v1, and the vertex pair (v1, v2) was not an edge in G.
    - The (n - 2)! permutations enumerated starting with (v1, v2) as its prefix would be a complete waste of effort.
    - It is better to stop the search after [v1, v2] and continue from [v1, v3].
        - This means to discard the search that finds [v1, v2] and instead try the next best option, which in this case is [v1, v3] as this is a valid edge in G.
- Pruning: The technique of abandoning a serach direction the instant we can establish that a given partial solution can't be extended into a full solution.
    - Also known as forward checking.

## Sudoku
- The state space will be the collection of open squares, each of which must eventually be filled with a digit.
- The candidates for open square (i, j) are integers from 1 to 9 that have no appeared in row i, column j, or the 3x3 sector containing (i, j).
    - We backtrack as soon as we are out of candidates for a square.
```py
def sudoku(board):
    """
    Non-optimised, simple sudoku solver.
    May fail TLE on leetcode in cases where most slots are empty spaces.
    """
    def is_valid(row, col, candidate):

        # Check column
        for it_col in range(9):
            if board[row][it_col] == candidate:
                return False
        
        # Check row
        for it_row in range(9):
            if board[it_row][col] == candidate:
                return False

        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[box_row + i][box_col + j] == candidate:
                    return False

        return True
    
    def backtrack():
        for row in range(9):
            for col in range(9):
                if board[row][col] == ".":                  # Empty space
                    for candidate in range(1, 10):
                        if is_valid(row, col, str(candidate)):
                            board[row][col] = str(candidate)
                            if backtrack():
                                return True
                            board[row][col] = "."           # Undo

                    return False                            # No valid solution
        return True                                         # No empty cells left means we solved the problem

    backtrack()
```

## Best-First Search
- Best-First search assigns a heuristic cost to every partial solution or frontier node.
- A priority queue is used to expand the node with the lowest heuristic value, which is the node closest to the goal.
- Chooses the best option for right now, making it a greedy algorithm.

## A* Heuristic
- Combines Best-First Search with Uniform-Cost Search or Dijkstras (UCS ~= Dijkstras).
    - Uniform-Cost Search (UCS): Expands path with the lowest total edge cost from the start.
- Using a lower bound on the cost of all partial solution extensions is stronger than just the cost of a current partial tour.
- f(n) = g(n) + h(n)
    - g(n): current path cost.
    - h(n): heuristic estimated cost from node n to goal.
    - f(n): estimated total cost of path through node n to the goal.
- Without h(n), the algorithm is essentially Dijkstra's algorith.

# Ch 10 - Dynamic Programming
- Algorithms for optimisation problems require proof that they always return the best possible solution.
    - Greedy algorithms that make the best local decision at each step are typically efficient, but don't guarantee global optimality.
    - Exhaustive search algorithms that try all possibilities and select the best always produces an optimal result, but have poor time complexity.
- DP allows us to design algorithms that systematically search all possibilities, while storing intermediate results to avoid recomputing.
- It is usually the right method for optimisation problems on combinatorial objects that have a left to right order among components.
    - Includes strings, rooted trees, polygons, integer sequences.

## Caching vs Computation
- DP is a tradeoff of space for time.
- In principle, caching can be used on any recursive algorithm, but storing partial results is useless for algorithms such as quicksort, backtracking and DFS because the values that are stored only get used once.
- Caching makes sense when the space of distinct parameter values is small enough that we can afford the cost of storage.

### Fibonacci
- Recursive algorithm takes O(1.6^n) time.
- We can cache results of each fibonacci computation F(k) in a tabled indexed by k (memoisation).
- This DP approach runs in O(n) time and requires O(n) space.
```py
def fib_memo(n, memo={}):
    if n in memo.keys():
        return memo[n]
    if n in [0, 1]:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]

print(fib_memo(10))     # Prints 55
```

### Binomial Coefficients
- n Choose k = number of ways to choose k things out of n possibilities.
    - nCk = n! / (k! * (n-k)!)
- In principle, you can compute them straight from factorials, but intermediate calculations can cause arithmetic overflow, even if the final coefficient fits within an integer.
- A more stable way is to use the recurrence relation implicit in the construction of Pascal's triangle.
    - Each number is the num of the two number sdirectly above it.
    - nCk = (n-1)C(k-1)+(n-1)Ck
```py
def binomial_memo(n, k, memo={}):
    if (n,k) in memo:
        return memo[(n,k)]
    
    if k == 0 or k == n:
        return 1

    memo[(n,k)] = binomial_memo(n-1,k-1,memo) + binomial_memo(n-1,k,memo)
    return memo[(n,k)]

print(binomial_memo(5, 2))      # Prints 10
```

## Approximate String Matching
- We already know the algorithms for exact string matching - finding where the pattern string P occurs as a substring of the text T.
- We want to search for the substring closest to a given pattern, to compensate for spelling errors.
- We first define a cost function telling us how far apart two strings are.
    - A reasonable distance measure reflects the number of changes that must be made to convert one string to another. There are 3 natural types of changes:
        1. Substitution: replace a single character in pattern P with a different character.
        2. Insertion: insert a single character into P to help it match text T.
        3. Deletion: delete a single character from P to help it match T.
    - Each operation has a cost of 1

### Edit Distance by Recursion
- The last character in the string must be matched, substituted, inserted, or deleted.
- Chopping charcters leaves a pair of smaller strings
    - Let i, j be indices of the last character of the relevant prefix of P and T.
    - There are 3 pairs of shorter strings after the last operation, corresponding to the strings after a match/substitution, insertion or deletion.
    - If we knew the cost of editing these 3 pairs of smaller strings, we could decide which options lead to the best solution and choose options accordingly.
- Let D[i,j] be the minimum number of differences between substrings P_1,...,P_n and T_1,...,T_j.
    - D[i,j] is the minimum of the three possible ways to extend smaller strings.
        1. If (Pi = Tj), then D[i-1, j-1], else D[i-1, j-1] + 1
            - This means we either match or substitute the ith and jth characters, depending on whether the tail characters are the same.
            - The cost of a single character substitution can be returned by a function match(Pi, Tj).
        2. D[i, j-1] + 1
            - This means there is an extra character in the text to account for, so we need to do an insertion.
            - The cost of a single character insertion can be returned by a function indel(Tj).
        3. D[i-1, j] + 1
            - This means there is an extra character to remove, so we need to do a deletion.
            - The cost of a single character deletion can be returned by a function indel(Pj).
```py
MATCH = 0
INSERT = 1
DELETE = 2

def indel(char):
    # Cost of insertion/deletion
    return 1    # Fixed cost

def match(c1, c2):
    # Cost of match or substitution
    return 0 if c1 == c2 else 1

def str_comp_r(s, t, i, j):
    if i == 0:
        return j    # Cost of inserting all remaining characters of t
    if j == 0:
        return i    # Cost of deleting all remaining characters of s
    
    # Defining option costs
    options = [0] * 3
    opt[MATCH] = str_comp_r(s, t, i-1, j-1) + match(s[i-1], t[j-1])
    opt[INSERT] = str_comp_r(s, t, i, j-1) + indel(t[k-1])
    opt[DELETE] = str_comp_r(s, t, i - 1, j) + indel(s[i-1])

    # Find the lowest cost option
    lowest_cost_opt = min(opt)
    return lowest_cost_opt

# Example Usage
s = "intention"
t = "execution"
result = str_comp_r(s, t, len(s), len(t))
print(f"Minimum edit distance: {result}")
```
- While this is a correct implementation, it has exponential time complexity because it recalculates values.
- At every position in the string, the recursion branches 3 ways, so it grows at a rate of at least 3^n.

**DP Implementation**
```py
MATCH = 0
INSERT = 1
DELETE = 2

def indel(char):
    # Cost of insertion/deletion
    return 1    # Fixed cost

def match(c1, c2):
    # Cost of match or substitution
    return 0 if c1 == c2 else 1

def str_comp_memo(s, t, i, j, memo):
    # Checking if the result is already cached
    if (i, j) in memo:
        return memo[(i, j)]

    if i == 0:
        return j    # Cost of inserting all remaining characters of t
    if j == 0:
        return i    # Cost of deleting all remaining characters of s
    
    # Recursive cases with memoisation
    options = [0] * 3
    opt[MATCH] = str_comp_r(s, t, i-1, j-1, memo) + match(s[i-1], t[j-1])
    opt[INSERT] = str_comp_r(s, t, i, j-1, memo) + indel(t[k-1])
    opt[DELETE] = str_comp_r(s, t, i - 1, j, memo) + indel(s[i-1])

    # Find lowest cost option and cache the result
    memo[(i, j)] = min(opt)
    return memo[(i, j)]

def string_compare(s, t):
    memo = {}
    return str_comp_memo(s, t, len(s), len(t), memo)

# Example Usage
s = "intention"
t = "execution"
result = str_comp_memo(s, t, len(s), len(t))
print(f"Minimum edit distance: {result}")
```
### Varieties of Edit Distance
- Substring matching: We want to find where a short pattern P best occurs within a long text T.
    - Without modification, the edit distance function considers the cost of deleting all characters in T that are not part of P. Starting a match in the middle of T also incurs additional costs from the prefix of T that doesn't align with P.
    - To make it works, we modify edit distance to allow:
        1. Flexible starting point
            - Instead of aligning P with start of T, we compute the edit distance for P against every possible substring of T.
            - The starting point of the match doesn't incur a cost penalty.
        2. Flexible goal state
            - In the regular edit distance, the goal state is at dp[len(P)][len(T)], which represents aligning P with the entirety of T.
            - Here, the goal state can occur at any position in T.
- Longest common subsequence (LCS): We want to find the longest scattered string of characters included in both strings, without changing the relative order.
    - Here, P can align anywhere within T, so the goal is to find the maximum LCS value over all positions of T.
    - A common subsequence is defined by all the identical character matches in an edit trace.
        - To maximise the number of such matches, we need to prevent the substitution of non-identical characters.
        - This way, the only way to remove a non-common subsequence is through insert/delete.
        - The minimum cost has the fewest insert/deletes, so it preserves the longest common substring.
- Maximum monotone subsequence: a numerical sequence is monotonically increasing if the ith element is at least as big as the (i-1)st element.
    - The maximum monotone subsequences seeks to delete the fewest number of elements from an input string S to leave a monotonically increasing subsequence.
    - This is just LCS, where the second string is the elements of S sorted in increasing order.

## Longest Increasing Subsequence
- There are 3 steps to solving problems using DP:
    1. Formulate the answer as a recurrence relation.
    2. Show that the number of different values taken on by your recurrence is bounded by a polynomial.
    3. Specify an evaluation order for the recurrence so the partial results you need are always available when you need them.
- Find the longest monotonically incresaing subsequence with a sequence of n numbers.
    1. DP table
        - Let dp[i][j] represent the length of the LCS of the first i characters of X and first j characters of Y.
    2. Recurrence relation
        - If characters match (X[i-1] == Y[j-1]):
            - dp[i][j] = dp[i-1][j-1] + 1
            - Include the matching character in the LCS.
        - If characters don't match:
            - dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            - Take the maximum LCS length by skipping one character from either X or Y.
    3. Base case
        - If either string is empty:
            - dp[i][0] = 0, dp[0][j] = 0
            - The LCS of an empty string with any string is 0.
    4. Final result
        - The value in dp[m][n] where m = len(X) and n = len(Y) is the length of the LCS of X and Y.
    - To reconstruct the actual sequence:
        - For each element s_i, we store its predecessor, the index p_i of the element that appears immediately before s_i in a longest incresaing sequence ending at s_i.
        - Then start from the last value in the longest sequence and follow the pointers back.
    - Time Complexity: O(mn) since we fill out the DP table.

## Unordered Partition/Subset Sum
- Subset sum problem: does there exist a subset S' of an input multiset of n positive integers S = {s1,...,sn} whose elements add up to a given target k.
- Main idea (top down approach):
    - Either the nth integer s_n is part of the subset adding up to k or it's not.
        - If it is, there must be a way to make a subset of the first n-1 elements of S adding up to k - sn.
        - If not, there may be a solution that doesn't use sn.
- Main idea (bottom up approach):
    - dp[i][k] is True if it is possible to achieve the sum k using the first i elements of nums.
    - The solution to the problem is dp[n][k] where n is the number of elements in nums.
- Time Complexity: Filling the table takes O(nk).

## Ordered Partition Problem
- Is it possible to divide a given array into two subsets such that the sum of elements in both subsets is equal?
- Main idea (bottom up approach):
    - dp[i][j]: boolean value indicating whether it is possible to achieve a sum j using the first i elements of nums.
    - Steps:
        1. Check feasability
            - Compute totalSum = sum(nums)
            - If totalSum % 2 != 0, return False
        2. Define the Dp table
            - dp[i][j]: true if a sum j can be achieved using the first i elements.
        3. Base case
            - dp[i][0] = true: a sum of 0 is always achievable by choosing no elements.
        4. Recurrence relation
            - If j < nums[i-1], exclude the current element.
                - dp[i][j] = dp[i-1][j]
            - Else, include or exclude the current element
                - dp[i][j] = dp[i-1][j] OR
                - dp[i][j] = dp[i-1][j-nums[i-1]]
        5. Final answer
            - Return dp[n][target], where target = totalSum / 2

## Parsing Context-Free Grammars
- Compilers identify whether a particular program is a legal expression in a particular language, and gives syntax errors if not.
- This requires a description of the language syntax using context free grammar.
    - Each rule or production of the grammar defines an interpretation for the named symbol on the left side of the rule as a sequence on the right side of the rule.
    - The right side is a combination of nonterminal and terminal symbols.
- Parsing a given text sequence S with a given CFG G is the algorithm problem of constructing a parse tree of rule substitutitons defining S as a single nonterminal symbol of G.

### Chomsky Normal Form (CNF):
- Determines whether a string w belongs to a languaged generated by a context-free grammar (CFG) in CNF.
- Grammar Format:
    - Every production must be either:
        - A -> BC (two nonterminals)
        - A -> a (single terminal)
    - If grammar is not in CNF, convert it first.
- DP Table Definition:
    - Build a 2D table where
        - i = starting index of substring (0-based)
        - j = length of substring
        - dp[i][j] = set of nonterminals that can generate substring w[i:i+j]
- Algorithm steps:
    1. Initialisation (length = 1 substring)
        - For each position i:
            - For every rule A -> w[i], add A to dp[i][1]
        - This fills the bottom row of the dp table.
    2. Build table for longer substrings
        - For substring lengths j = 2 ... n:
        - For each starting position i:
        - For every possible split point k:
            - NOTE: first part has length k, second part length j - k
            - For each production A -> BC:
                - If B \in dp[i][k] and C \in dp[i+k][j-k], add A to dp[i][j]
        - This is the core recurrence
    3. Acceptance Condition
        - The string w is in the language iff:
            - Start symbol S \in dp[0][n]
            - where n is the length of w
- Time Complexity: O(n^3 * |G|)
- Space Complexity: O(n^2)

## Limitations of DP: TSP
- Longest simple path: what is the most expensive path from s to t that does not visit any vertex more than once.

### When is DP correct?
- Only as correct as the recurrence relations they are based on.
- Can be applied to any problem that obeys the principle of optimality.
    - This means partial solutions can be optimally extended given the state after the partial solution, instead of the speicifics of the partial solution itself.
    - Future decisions are made based on the consequences of previous decisions, not the actual decision themselves.

### When is DP efficient?
- The runtime of a DP algorithm is a function of two components:
    1. The number of partial solutions we need to track.
    2. How long it takes to evaluate each partial solution.
- Generally, the first issue is the main concern.

# Ch 11 - NP-Completeness
- P: Problems that can be solved in polynomial time.
- NP: Problems whose solutions can be verified in polynomial time.
- P \eqSubset NP: every problem in P is in NP, but we haven't proven that vice versa also true.
    - If you can solve the problem in polynomial time, then you must've verified that it was correct in polynomial time as well.

## Problems and Reductions
- If a problem X can be reduced to an instance of problem Y, then it means that problem Y is at least as ahrd as problem X.
- Decision problems: Does there exist an object satisfying some given properties?
- Optimisation problems: What is the size of the biggest/smallest such object?
- Search: Find the biggest/smallest such object
- Decision <=p Optimisation: run optimisation algorithm and check if output >= k.
- Optimisation <=p Search: run search algorithm and return size of the input.
- NP-Complete: A problem is NP-complete if:
    1. It is NP Hard: every problem in NP can reduce to it.
    2. Is in NP: can be verified in polynomial time.

### Convex Hull (*)
- A polygon is convex is the straight line segment drawn between any two points inside polygon P lies completely within the polygon.
- A convex hull provides a very useful way to give strutcure to a point set.

## Elementary Hardness Reductions

### Hamilton Cycle
- Input: Unweighted graph G.
- Output: Does there exist a simple tour that visits each vertex of G without repetition?
    - i.e each vertex is visited exactly one time.

### Independent Set and Vertex Cover
- Vertex Cover:
    - Input: Graph G = (V, E) and integer k <= |V|.
    - Output: Is there a subset S of at most k vertices such that every e \in E contains at least on vertex in S.
- Independent Set:
    - Input: Graph G = (V, E) and integer k <= |V|.
    - Output: Does there exist an independent set of k vertices in G?
        - Independent set: A set of vertices such that no two vertices are directly connected by a single edge.
- Reduction:
    - If S is the vertex cover of G, the remaining vertices S - V must form an independent set, for if an edge had both vertices in S - V, then S could not be a vertex cover.
```
VertexCover(G, k):
    G' = G
    k' = |V| - k
    Return IndependentSet(G', k')
```

### Clique
- A clique is a complete subgraph where each vertex pair has an edge between them.
- Input: Graph G = (V, E) and integer k <= |V|.
- Output: Does the graph contain a clique of k vertices?
```
IndependentSet(G, k):
    Construct graph G' = (V', E') where V' = V and:
        for all (i, j) not in E, add (i, j) to E'           # Only add vertex pairs with edge between them
    Return the answer to Clique(G', k)
```

### Satisfiability
- Input: Set of Boolean variables V and set of clauses C over V.
- Output: Does there exist a satisfying truth assignment for C?
    - A way to set the variables v1, ..., vn true or false so that each clause contains at least one true literal.
- 3-Satisfiability:
    - Input: Collection of clauses C where each clause contains exactly 3 literals (3 variables in every clause), over a set of boolean variables V.
    - Output: Is there a truth assignment to V such that each clause is satisfied?
```
Take Variables x1, x2, x3
And clauses:
    C1 = (x1 ∨ ¬x2 ∨ x3)
    C2 = (¬x1 ∨ ¬x3)
    C3 = (x2 ∨ x3)

x1 = x3 = True, x2 = False
Is unsatisfiable as C2 = (FALSE or FALSE) = False.

On the other hand,
x2 = x3 = True, x1 = False
All clauses are true, meaning this equation is satisfiable.
```

## P vs NP
- Verification: Checking if a given solution to a problem is correct.
- Discovery: Finding the actual solution to a problem.
- P: Exclusive club for algorithm problems where there exists a polynomial-time algorithm to solve it from scratch.
    - Shortest path, minimum spanning tree, movie scheduling problem.
    - P stands for polynomial time.
    - Polynomial time means the algorithm is bounded by some time complexity.
    - Variable is the base (e.g n^k where k is a constant).
- NP: Non polynomial.
    - Running time is not necessarily bounded.
    - Exponential time, variable is the exponent (e.g k^n where k is a constant).

## Dealing with NP-Complete Problems
- An NP-Complete problem is a decision problem (yes/no answer) that can be verified in polynomial time, and for which every othe rNP problem can be reduced in polynomial time.
    - If we find polynomial-time solution for one NP-complete problem, you could solve all problems in NP in polynomial time.
- We usually want to find a program to solve a problem of interest, even if we know that it won't be optimal in the worst case.
- There are 3 options:
    1. Algorithms fast in the average case: e.g backtracking algorithms with pruning.
    2. Heuristics: methods like simulated annealing or greedy approaches can be used toq uickly find a solution, although there is no guarantee it is the best solution.
    3. Approximation algorithms: NP completeness only states that it is hard to get close to the answer. With clever problem specific heuristics, we can probably get close to the optimal answer on all possible instances.

### Approximating Vertex Cover
- A simple procedure can be used to find a cover that is at most twice as large as the optimal cover.
```
VertexCover(G = (V, E)):
    While (E != null):
        Pick any edge (u, v) in E
        Add both u and v to vertex cover
        Delete all edges from E that are incident to either u or v
```

### Maximum Acyclic Subgraph
- Directed acyclic graphs are easier to work with than general digraphs.
- Input: Directed graph G = (V, E).
- Output: Find largest possible subset E' \in E such that G' = (V, E') is acyclic.
    - Construct any permutation of the vertices, and interpret it as a left-right ordering.
    - Now, some of the edges point left to right, while the remainder point right to left.
    - One of thes two edge subsets must be at least as large as the other, meaning it contains at least half the edges.
        - Furthermore, these two edge subsets must be acyclic.
        - This must contain at least half the edges of the optimal solution.

### Set Cover
- Input: Collection of subsets S = {s1, ..., sm} of the universal set U = {1, ..., n}.
- Output: What is the smallest subset T of S whose union equals the universal set?
```
SetCover(s):
    While (U != null):
        Find subset Si with largest intersection with U     # Find best current subset
        Select Si as set cover                              # Set setcover
        U = U - Si                                          # Remove claimed values and iterate
```

# Ch 12 - Dealing with Hard Problems

## TSP
- Considering a minimum spanning tree, we know that the minimum weight of the optimal path for TSP is bounded by the total weight of the MST.
- If exploring the MST through DFS, we get costs at most twice the total cost of the tree. This is because we explore each edge twice.
- If we instead take a direct path to the next unvisited vertex at each step, it is must faster.
    - We construct a new "shortcut" tour in O(n + m), where n = number of vertex and m = number of edges.
    - This always has weight at most twice of the optimal TSP tour of G.

### Christofides Heuristic
- Eulerian cycle: Circuit in graph G traversing each edge exactly once.
    - Circuit: Walk with no repeated edges.
    - Cycle: Walk where no vertices are repeated except the starting and ending vertex.
    - Can easily test if a connected graph contains this cycle, by just checking if every vertex has an even edgree.
- We can use this to construct a new multigraph which recruits every edge in the MST twice.
- Hence we can construct a TSP tour with a cost at most twice that of the optimal tour.
- If we can find a cheaper way to ensure all vertices are of even degree, we may find a better approximation for TSP.
- Lowest cost perfect matching: Every vertex must appear in exactly one matching edge.
- Christofides heuristic constructs a multigraph M consisting of the minimum spanning tree of G plus the minimum weight set of matching edge between odd-degree vertices in this tree.
    - Thus, M is an Eularian graph, and contains an Eularian cycle that can be shortcut to build a TSP tour of weight at most M.
- Using this heuristic allows us to find a solution of at most 3/2 times the weight of the optimal tour.

## Heuristic Searches
**2 Core Components**:
- Solution Candidate Representation: Complete yet concise description of possible solutions for a problem, just like in backtracking.
    - For TSP, solution search space contains (n-1)! elements, precisely every possible circular permutation of all vertices.
    - Candidate solution can be represented using an array S of n - 1 vertices, where Si defines the (i+1)st vertex on the tour starting from v1.
- Cost Function: Search methods need a cost or evaluation function to assess the quality of each possible solution
    - Search heuristic identifies element with the best score.
    - For TSP, cost function for evaluating candidate solution S just sums the weight of all edges, where S0 and Sn both denote the starting vertex.

### Random Sampling
- AKA Monte Carlo method.
- Repeatedly construct random solutions and evaluate them, stopping once we get a good enough solution or (more likely) when we get tired of waiting.
- We report the best solution we found.
- True random sampling requires each elements in the solution space having an equally likely probability of being the next selected candidate.
- Useful when:
    - There is a large proportion of acceptable solutions, such as finding hay in a haystack.
    - Finding prime numbers, as roughly one in every ln(n) integers are prime, so only need to choose and select a modest number of random samples.
    - No coherence in the solution space: Random sampling is appropriate when you have no sense if you are getting closer to a solution.

### Local Search
- Scans the neighbourhood around elements in the search space.
- Start from some arbitrary element of the solution space, then scan the neighbourhood looking for a favourable transition to take.
- In Hill Climbing, this can take us to local maxima instead of global maxima, as we stop when there are no more favourable transitions nearby.
- Useful when:
    - Great coherence in search sapce: Hill climbing works best with exactly one hill, meaning you end up at the global maximum no matter where you start.
    - When the cost of incremental evaluation is much cheaper than global evaluation.
- Depending on search space, there may not be much we can do after finding the local optimum.
    - For example, random restarts might not be very helpful if there is many low hills.

### Simulated Annealing
- Heuristic search procedure that allows occasional transitions leading to more expensive (heuristically inferior) solutions.
- Assigns a temperature T that decreases over time.
    - A higher temperature allows more random moves, meaning it is more likely to select a worse move.
    - As T cools down (decreases), worse moves become less likely, and better moves become more likely.
```
# Note: C = cost function
SimulatedAnnealing()
    Create initial solution s
    Initialise temperature T
    repeat
        for i = 1 to iteration-length do
            Randomly select neighbour of s to be si
            If (C(s) >= C(si)) then s = si
            else if (e^(C(s)-C(si))/(kB * T) > random[0, 1]) then s = si
        Reduce temperature T
    until (no change in C(s))
    Return s
```
- Cooling function can be designed however the user prefers.
- As an example, we can start with T1 = 1, and decrement it by:
    - Ti = alpha * T(i-1), where 0.8 <= alpha <= 0.99

#### Applications of Simulated Annealing:
- Maximum Cut
    - Seeks to partition the vertices of weighted graph G into sets V1 and V2 to maximise the weight (or number) of edges with one vertex in each set.
    - NP-Complete.
- Independent Set
    - An independent set of graph G is a buset of vertices S such that there is no edge with both endpoints in S.
    - The maximum independent set is the largest vertex set that induces an empty (edgeless) subgraph.
- Circuit Board Placement
    - Sometimes when designing printed circuit boards, we face problems with positioning modules (like in integrated circuits).
    - Desired criteria in a layout may include:
        1. Minmising the area or optimising the aspect ratio of the board so that it fits in a particular space.
        2. Minimising the total or longest wire length in connecting the components.
    - This problem is representative of the type of messy, multicriterion optimisation problems in which simulated annealing is suited.

## Genetic Algorithms
- Draws inspiration from evolution and natural selection.
- Maintains a population of solution candidates for the given problem.
- Fitness function evaluates how good a candidate is for this problem.
- Selection, crossover, mutation generate new candidates by combining and randomly altering high fitness solutions.
- We iterate over generations until the population converges or a good enough solution is found.

## Discrete Fourier Transform
- Takes a list of numbers (a signal) and expresses it as a combination of sine and cosine waves of different frequencies.
- Tells you how much of each frequency is present in the signal.
- Mathematically it converts data from a time domain into a frequency domain.
- The formula uses complex numbers and costs O(n^2) time.
    - This can be sped up using FFTs as explained below.

### Faster Fourier Transform:
- FTT Computes the discrete Fourier transform (DFT) in O(nlog(n)) instead of O(n^2).
- Divide and conquer: split the input into even and odd indexed parts, comupte their DFTs recursively, then combine.
- Use "twiddle factors" (complex roots of unity) to efficiently recombine the sub-results.
- Widely used for signal processing, convultion, audio/image compression, and many scientific computations.

## Shor's Algorithm for Integer Factorisation
- Uses quantum computing to find the period of a modular function.
- Converts integer factorisation into a period-finding problem.
- Quantum Foruier transform finds this period exponentially faster than classical methods.
- Uses the period to compute the non-trivial factors of the target number.