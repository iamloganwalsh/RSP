# Minimum Spanning Tree
- A minimum spanning tree is a way to transform a graph into only keeping the lowest weighted edges while still being able to reach every vertex from every vertex.
- The MST is the smallest connected subgraph of some graph G which has the minimum possible total weights (while still being connected).
- For directed graphs, we can find the MST using two algorithms:
    - Prims: Grow the MST by starting from any node and repeatedly adding the cheapest edge that connects the tree to a new node.
        - Can do this easily using a min heap to sort edges by weight and a visited hashtable to only visit new nodes.
    - Kruskals: Start by having all vertices as disconnected graphs with no edges. Build the MST by sorting all edges by weight and adding them in order, skipping any that would form a cycle.