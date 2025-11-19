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
    - Graph's diameter: The largest shortest-path distance over all pairs of vertices.
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