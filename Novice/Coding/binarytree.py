# BinaryTree also acts as a node
class BinaryTree:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        
    def is_full_node(self):
        if self.left and self.right or not self.left and not self.right:
            return True
        return False

    def insert_left(self, new_data):
        if self.left == None:
            self.left = BinaryTree(new_data)
        else:
            tree = BinaryTree(new_data, left=self.left)
            self.left = tree

    def insert_right(self, new_data):
        if self.right == None:
            self.right = BinaryTree(new_data)
        else:
            tree = BinaryTree(new_data, right=self.right)
            self.right = tree
