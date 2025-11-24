# Dijkstra's Algorithm
- Dijkstra's algorithm finds the shortest path between a source node and all other nodes in a graph with non-negative edge weights.
- Maintains a priority queue (using a min-heap) of nodes ordered by their current best-known distance.
- When a node is popped, if its distance is larger than an already-recorded distance, the entry is discarded.
- Otherwise, we update the neighbours distance and push it back to the priority queue.

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

            if dist < distances[neighbour]:                     # Enqueue new nodes if its shorter than the known path
                distances[neighbour] = dist
                heapq.heappush(pq, (dist, neighbour))
    
    return distances
```

Time Complexity: O((V+E)log(V)).<br>
    - V nodes and need to process each node in log(V) time, so this step takes O(Vlog(V)).
    - For each node, the algorithm checks its neighbours and updates distance. Each edge is checked once during the algorithm, which takes O(E).
    - Worst case (V^2log(V)) if G is a dense graph.
Axuillary Space: O(V+E).
    - Seen vertices and heapq edges