import numpy as np


class Graph:

    def __init__(self):
        self.operation = []
        self.placeholders = []
        self.variables = []
        self.constants = []
        

class Operation:
    
    def __init__(self, input_nodes=None):
        self.input_nodes = input_nodes
        self.output = None
    
    def forward(self):
        pass
        
    def backward(self):
        pass
        
        
class BinaryOperation(Operation):
    
    def __init__(self, a, b):
        super().__init__([a, b])
        

class add(BinaryOperation):
    
    def forward(self, a, b):
        return a + b
        
    def backward(self, upstream_grad):
        pass
        
        
class multiply(BinaryOperation):
    
    def forward(self, a, b):
        return a * b
        
    def backward(self, upstream_grad):
        pass
        
        
class divide(BinaryOperation):
    
    def forward(self, a, b):
        return np.true_divide(a, b)
        
    def backward(self, upstream_grad):
        pass
        

class matmul(BinaryOperation):

    def forward(self, a, b):
        return a.dot(b)
        
    def backward(self, upstream_grad):
        pass
        
        
class Placeholder:
    
    def __init__(self):
        self.value = None
        

class Constant:
    
    def __init__(self):
        self.__value = value
        
    @property
    def value(self):
        return self.__value
        
    @value.setter
    def value(self, value):
        raise ValueError("Cannot reassign value.")
        
        
class Variable:

    def __init__(self, initial_value=None):
        self.value = initial_value

        
def topology_sort(operation):
    ordering = []
    visited_nodes = set()
    
    def recursive_helper(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                if input_node not in visited_nodes:
                    recursive_helper(input_node)
                    
        visited_nodes.add(node)
        ordering.append(node)
    
    recursive_helper(operation)
    return ordering
    

class Session:

    def run(self, operation, feed_dict={}):
        nodes_sorted = topology_sort(operation)
        
        for node in nodes_sorted:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable or type(node) == Constant:
                node.output = node.value
            else:
                inputs = [node.output for node in node.input_nodes]
                node.output = node.forward(*inputs)
    
        return operation.output
