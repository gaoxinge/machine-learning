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
