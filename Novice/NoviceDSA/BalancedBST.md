# Balanced BST's
- A balanced BST is a binary search tree whose height difference between the left and right subtrees of any node is at most one.

## AVL
- Self balancing BST.
- Balance factor = left subtree height - right subtree height.
- For a balanced tree, for every node: -1 <= balance factor <= 1.
    - Each node has some value that represents the difference in height between the left and right subtrees.
    - An AVL tree requires that every node's value is in the balance factor range shown above.
- Uses rotations to restore balance in O(1) time, keeping the overall time complexity at O(log(n)).
    - Rotations used in insertion and deletion.
- Search, insert and delete are all O(log(n)) opeartions (like normal BST), and then the rotations to balance the tree are O(1) each.

### AVL - Insert
- A new key is placed in the correct position like a normal BST.
- The AVL tree rotates to ensure the balance factor is correct.
- Can use multiple rotations as long as we can balance the tree.

### AVL - Deletion
- Similar process to insert.
- Firstly, perform delete like in a normal BST.
    1. Find the node
    2. Replace with inorder successor, inorder predecessor, or null (if no children).
    3. Rotate nodes that violate the balance factor to balance the tree.

## Red-Black Trees
- Every node gets coloured either red or black.
    - The root of the tree is always black.
    - Red property: red nodes cannot have red children.
    - Black property: every path from a node to its descendant null nodes (leaves) has the same number of black nodes.
    - All leaf nodes are black.
- Height is at max O(log(n)).
- Colours are used to maintain balance during insertions and deletions, ensuring efficient data retrieval and manipulation.

### RB - Insert
1. Insert like a normal BST.
2. Fix violations:
    - If the parent of the new node is black, no properties are violated.
    - If the parent is red, the tree might violate the red property, requiring fixes.
- Case 1: Uncle is Red: recolor the parent and uncle to black, and the grandparent to red. Then move up the tree to check further violates.
- Case 2: Uncle is Black: 
    - Subcase 2.1: Node is a right child: perform left rotation on the parent.
    - Subcase 2.2: Node is a left child: perform a right rotation on the grandparent and recolour appropriately.

### RB - Deletion
1. Delete like a normal BST.
2. Fix Double Black:
    - If a black node is deleted, a "double black" condition might arise, which requires specific fixes.
- Case 1: Sibling is Red: Rotate the aprent and recolour the sibling and parent.
- Case 2: Sibling is Black:
    - Subcase 2.1: Sibling's children are black: recolour the sibling and propagate the double back upwards.
    - Subcase 2.1: At least one of the sibling's children is red:
        - If the sibling's far child is red: perform a rotation on the parent and sibling, and recolour appropriately.
        - If the sibling's near child is red: rotate the sibling and its child, then handle as above.