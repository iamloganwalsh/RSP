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
