# Dijkstra's Algorithm
- Dijkstra's algorithm finds the shortest path between a source node and all other nodes in a graph with non-negative edge weights.
- Maintains a priority queue (using a min-heap) of nodes ordered by their current best-known distance.
- When a node is popped, if its distance is larger than an already-recorded distance, the entry is discarded.
- Otherwise, we update the neighbours distance and push it back to the priority queue.