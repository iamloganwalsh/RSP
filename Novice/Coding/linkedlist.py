class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self, head=None):
        self.head = head
        self.count = 1 if head else 0

    def add_node(self, new_node):
        if not self.head:
            self.head = new_node
            self.count = 1
            return

        # Traverse linked list
        curr = self.head
        while curr.next:
            curr = curr.next
        curr.next = new_node
        self.count += 1

    # Assumes unique node values
    def delete_node_by_value(self, node_value):
        if not self.head:
            return False

        # Check head
        if self.head.value == node_value:
            self.head = self.head.next
            self.count -= 1
            return True
        
        if not self.head.next:
            return False

        # Traverse linked list
        prev = self.head
        curr = self.head.next
        while curr:
            if curr.value == node_value:
                self.count -= 1
                prev.next = curr.next
                return True
            prev = curr
            curr = curr.next

    
    def is_empty(self):
        return self.count == 0
    
    def size(self):
        return self.count
    
    def clear(self):
        self.head = None
        self.count = 0
        
    def __str__(self):
        str_builder = ""

        curr = self.head
        while curr:
            str_builder += curr.value + " -> "
            curr = curr.next
        
        return str_builder[:-4] # -4 removes the final arrow

        
        
    
if __name__ == "__main__":
    # Test 1: Empty list initialization
    empty_list = LinkedList()
    print(f"Empty list: {empty_list}")
    print(f"Is empty: {empty_list.is_empty()}")
    
    # Test 2: Add to empty list
    empty_list.add_node(Node("A"))
    print(f"After adding to empty: {empty_list}")
    
    # Test 3: Normal list
    ll = LinkedList(Node("A"))
    ll.add_node(Node("B"))
    ll.add_node(Node("C"))
    print(f"\nNormal list: {ll}")
    print(f"Size: {ll.size()}")
    
    # Test 4: Delete head
    ll.delete_node_by_value("A")
    print(f"After deleting head: {ll}")
    
    # Test 5: Delete last node
    ll.delete_node_by_value("C")
    print(f"After deleting last: {ll}")
    
    # Test 6: Delete non-existent node
    result = ll.delete_node_by_value("Z")
    print(f"Delete non-existent node returned: {result}")
    
    # Test 7: Delete from single-node list
    ll.delete_node_by_value("B")
    print(f"After deleting only node: {ll}")
    print(f"Is empty: {ll.is_empty()}")
    
    # Test 8: Delete from empty list
    result = ll.delete_node_by_value("X")
    print(f"Delete from empty list returned: {result}")
    
    # Test 9: Clear method
    ll2 = LinkedList(Node(1))
    ll2.add_node(Node(2))
    ll2.clear()
    print(f"\nAfter clear: {ll2}")
