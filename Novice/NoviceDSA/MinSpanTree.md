# Minimum Spanning Tree
- A minimum spanning tree is a way to transform a graph into only keeping the lowest weighted edges while still being able to reach every vertex from every vertex.
- The MST is the smallest connected subgraph of some graph G which has the minimum possible total weights (while still being connected).
- For directed graphs, we can find the MST using two algorithms:
    - Prims: Grow the MST by starting from any node and repeatedly adding the cheapest edge that connects the tree to a new node.
        - Can do this easily using a min heap to sort edges by weight and a visited hashtable to only visit new nodes.
        - Time Complexity: O(E+Vlog(V)). (Fibonacci heap)
            - Adding edges to the PQ takes O(Elog(V)).
            - Extracting minimum edge takes O(log(V)), and we do it V times for a total of O(Vlog(V)).
    - Kruskals: Start by having all vertices as disconnected graphs with no edges. Build the MST by sorting all edges by weight and adding them in order, skipping any that would form a cycle.
        - Time Complexity: O(Elog(E)).
            - Sorting edges takes O(Elog(E)).
            - The while loop runs at most E times.

## KRUSKALS > PRIMS
- Better for sparse graphs
    - Sorts edges and processes one by one
    - Union find so can prevent cyles (if there is a common parent between two vertices we don't add the edge)
    - E ~= V
    - Union find is extremely fast (O(a(n))), and sorting ~V edges is cheap
    - More overhead from Prims pushing to PQ

## PRIMS > KRUSKALS
- Better for dense graphs
    - Grows MST one vertex at a time
    - Uses PQ to track candidate edges
    - E ~= V^2
    - Plugging into equations:
        - Kruskals becomes (V^2log(V^2))
        - Prims becomes (V^2 + Vlog(V))
        - V^2 < V^2log(V^2) so prims is faster here

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

        # Add valid edges to the MST
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