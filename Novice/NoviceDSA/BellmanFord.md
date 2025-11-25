### Bellman-Ford Algorithm
- Single source shortest path algorithm.
- Given a weighted graph G = (V, E) and a source vertex start, the task is to compute shortest distance from the **starting vertex** to all other vertices.
    - If a negative cycle occurs, return None
    - If a vertex is unreachable, the distance should be infinite
- Time Complexity: O(V * E).
- Principle of Relaxation of Edges: Relaxation means updating the shortest distance to a node if a shorter path is found through another node.
```py
import math

def bellman_ford(num_vertices, edges, start):

    # Initialise distance
    dist = [math.inf] * num_vertices
    dist[start] = 0

    # Relax all edges first, finding shortest paths
    for i in range(num_vertices - 1):
        for edge in edges:
            u, v, weight = edge

            # Relax edges (replace with shorter path)
            if dist[u] != math.inf and dist[u] + weight < dist[v]:

                dist[v] = dist[u] + weight

    # Checking negative cycles (Vth iteration)
    # Final iteration does not update, only checks for negative cycles
    # If the distance can still be improved on the final iteration, a negative cycle must exist
    # This is because legitimate shortest paths cannot have more than V-1 edges, so any improvement has to be due to an infinitely traversable negative cycle
    for edge in edges:
        u, v, weight = edge
        if dist[u] != math.inf and dist[u] + weight < dist[v]:
            return [None]
    
    return dist
```

Time Complexity: O(V * E).
- We check each vertex and edge exactly one time.
Auxillary space: O(V).
- We store the distance to each vertex.