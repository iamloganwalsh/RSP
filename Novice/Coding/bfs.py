from binarytree import BinaryTree
from collections import deque

def bfs(node, value):
    if not node:
        return False

    queue = deque([node])

    while queue:
        curr = queue.popleft()

        if curr.data == value:
            return True
        
        if curr.left: queue.append(curr.left)
        if curr.right: queue.append(curr.right)

    return False


if __name__ == "__main__":
    # Create a binary tree:
    #        5
    #       / \
    #      3   8
    #     / \   \
    #    1   4   9
    
    # Build the tree carefully
    root = BinaryTree(5)
    
    # Add left subtree
    root.left = BinaryTree(3)
    root.left.left = BinaryTree(1)
    root.left.right = BinaryTree(4)
    
    # Add right subtree
    root.right = BinaryTree(8)
    root.right.right = BinaryTree(9)
    
    # Test cases
    print("=== DFS Search Tests ===")
    print(f"Search for 5 (root): {bfs(root, 5)}")           # True
    print(f"Search for 3 (left child): {bfs(root, 3)}")     # True
    print(f"Search for 9 (right leaf): {bfs(root, 9)}")     # True
    print(f"Search for 1 (left leaf): {bfs(root, 1)}")      # True
    print(f"Search for 4 (middle): {bfs(root, 4)}")         # True
    print(f"Search for 10 (not in tree): {bfs(root, 10)}")  # False
    print(f"Search for 7 (not in tree): {bfs(root, 7)}")    # False
    
    # Edge case: empty tree
    print(f"\nSearch in None tree: {bfs(None, 5)}")         # False
    
    # Edge case: single node tree
    single = BinaryTree(42)
    print(f"Search for 42 in single node: {bfs(single, 42)}")  # True
    print(f"Search for 99 in single node: {bfs(single, 99)}")  # False
