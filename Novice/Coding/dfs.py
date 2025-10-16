from binarytree import BinaryTree

def dfs(node, value):
    if node is None:
        return False
    
    if node.data == value:
        return True
    
    return dfs(node.left, value) or dfs(node.right, value)


# Test Example
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
    print(f"Search for 5 (root): {dfs(root, 5)}")           # True
    print(f"Search for 3 (left child): {dfs(root, 3)}")     # True
    print(f"Search for 9 (right leaf): {dfs(root, 9)}")     # True
    print(f"Search for 1 (left leaf): {dfs(root, 1)}")      # True
    print(f"Search for 4 (middle): {dfs(root, 4)}")         # True
    print(f"Search for 10 (not in tree): {dfs(root, 10)}")  # False
    print(f"Search for 7 (not in tree): {dfs(root, 7)}")    # False
    
    # Edge case: empty tree
    print(f"\nSearch in None tree: {dfs(None, 5)}")         # False
    
    # Edge case: single node tree
    single = BinaryTree(42)
    print(f"Search for 42 in single node: {dfs(single, 42)}")  # True
    print(f"Search for 99 in single node: {dfs(single, 99)}")  # False
