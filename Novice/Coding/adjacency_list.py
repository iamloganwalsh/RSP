from collections import defaultdict

# Assuming only one directed edge is allowed 
#   (i.e node 0 can only directly point to node 1 through a single edge)
class DirectedGraph:
    def __init__(self):
        # Using a dictionary to map edges for adjacency list
        # Using Set to ensure only one edge per direction between two nodes
        self.adjacency_list = defaultdict(set)
        
        # Tracking existant nodes
        self.seen = set()

    def add_edge(self, u, v):
        # u -> v
        self.adjacency_list[u].add(v)
        self.seen.add(u)
        self.seen.add(v)

    def remove_edge(self, u, v):
        try:
            self.adjacency_list[u].remove(v)
        except:
            print(f"Edge does not exist between node {u} and node {v}")

    def __str__(self):
        str_builder = ""
        for node in self.seen:
            value = self.adjacency_list[node]
            str_builder += f"{node}: {' '.join(map(str, value))}\n"

        return str_builder[:-1] # Remove last empty line (\n)
    
# Assuming only one edge between nodes
class UndirectedGraph:
    def __init__(self):
        # Using a dictionary to map edges for adjacency list
        # Using Set to ensure only one edge per direction between two nodes
        self.adjacency_list = defaultdict(set)
        
        # Tracking existant nodes
        self.seen = set()

    def add_edge(self, u, v):
        self.adjacency_list[u].add(v)
        self.adjacency_list[v].add(u)
        self.seen.add(u)
        self.seen.add(v)

    def remove_edge(self, u, v):
        try:
            self.adjacency_list[u].remove(v)
            self.adjacency_list[v].remove(u)
        except:
            print(f"Edge does not exist between node {u} and node {v}")

    def __str__(self):
        str_builder = ""
        for node in self.seen:
            value = self.adjacency_list[node]
            str_builder += f"{node}: {' '.join(map(str, value))}\n"

        return str_builder[:-1] # Remove last empty line (\n)
    

# Test Cases
### Copied schema from ./dg_example.png and ./udg_example.jpg
if __name__ == "__main__":
    print("Directed Graph")
    dg = DirectedGraph()
    dg.add_edge(1, 2)
    dg.add_edge(1, 3)
    dg.add_edge(3, 2)
    dg.add_edge(3, 4)
    dg.add_edge(4, 3)
    print(dg, "\n")

    print("Undirected Graph")
    udg = UndirectedGraph()
    udg.add_edge(6, 4)
    udg.add_edge(4, 3)
    udg.add_edge(4, 5)
    udg.add_edge(3, 2)
    udg.add_edge(5, 2)
    udg.add_edge(5, 1)
    udg.add_edge(2, 1)
    print(udg)