# Union Find
- A collection of disjoint sets, in which no element can be found in more than one set.
- The disjoint set DS is used to store such sets, with the following operations:
    - Union(set1, set2): Merge two disjoint sets.
    - Find(element): Find's the "representative" of a disjoint set using the find operation.
    - Connected(element1, element2): Find if two elements exist in a set.

**General Description**: This DS maintains a collection of disjoint sets with a designated "leader" or "representative" for each set. Union find is useful for checking if a relationship exists between two elements, such as does person x know person y through direct or indirect friendships.

**Python Code**
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
            self.parent[x] = self.find(self.parent[x])      # Recursively find their parent (Breaks previous links so it only connects to the representative)
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
Time Complexity (PC & RANK): O(a(n)) where a(n) is the inverse ackermann function.<br>
    - Time Complexity (DEFAULT): O(log(n)) or O(n) worst case.
Auxillary Complexity: O(n).
- Store n parents.
- Store n ranks.

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

Time Complexity: O(nlog(n))
- Building heap: O(n) using bottom-up construction
- Heapifying: O(log(n)) per removal because we have to maintain the heap property (percolating down)
- Total: O(nlog(n)) building heap * performing n heapify operations
Auxillary Space: O(log(n)) for recursion or O(1) for iteration

### Heap Init & Insert
- Sequentially add each value to the heap, percolating on each iteration to ensure integrity.
- O(log(n))

### Heap Delete
- When we pop the root (i.e the max or min of the heap), we replace with the right-most (last) child.
- Iteratively percolate down to ensure heap integrity.
- O(log(n))