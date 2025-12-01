# Topological Sort
- Linear ordering of vertices in a DAG (directed acyclic graph) where for every directed edge (u, v), u comes before v in every ordering.
- Only possible in DAG because cycles create circular dependency, meaning no valid order could exist.
- Useful for scheduling tasks, resolving dependencies, determining valid sequences for operations.

## Using DFS:
- For each vertex, push it onto a stack only after visiting adjacent vertices.
- This means each vertex appears after all its neighbouring vertices.
    - E.g A -> B -> C
    - Stack: [C, B, A] ->
- Reversing the stack (or popping its elements) gives the topological ordering of the graph.

## Using BFS (Kahn's Algorithm):
- Repeatedly select vertices with in-degree zero (no dependencies / incoming edges), add them to the result, and reduce the in-degree of their adjacent vertices (simulating removing the edge).
- This continues until all vertices are processed, producing a valid linear ordering of a DAG.
1. Compute indegrees
2. Add nodes with indegree 0 to queue
3. BFS while decrementing indegree

**GeeksforGeeks Kahn's Implementation**
```py
def topoSort(adj):
    n = len(adj)
    indegree = [0] * n
    res = []
    queue = deque()

    # Compute indegrees
    for i in range(n):
        for next_node in adj[i]:
            indegree[next_node] += 1
            
    # Add all nodes with indegree 0 
    # into the queue
    for i in range(n):
        if indegree[i] == 0:
            queue.append(i)

    # Kahn’s Algorithm
    while queue:
        top = queue.popleft()
        res.append(top)
        for next_node in adj[top]:
            indegree[next_node] -= 1
            if indegree[next_node] == 0:
                queue.append(next_node)

    return res

def addEdge(adj, u, v):
    adj[u].append(v)
#Driver Code Starts

# Example
n = 6
adj = [[] for _ in range(n)]

addEdge(adj, 0, 1)
addEdge(adj, 1, 2)
addEdge(adj, 2, 3)
addEdge(adj, 4, 5)
addEdge(adj, 5, 1)
addEdge(adj, 5, 2)

res = topoSort(adj)
print(*res)
#Driver Code Ends
```