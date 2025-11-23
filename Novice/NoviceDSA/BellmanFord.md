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