
"""Recursive syntactic workspace stack for ROSE simulations."""
import numpy as np

class SyntaxWorkspace:
    def __init__(self, capacity=5):
        self.stack=[]
        self.capacity=capacity
    def push(self, item):
        if len(self.stack)>=self.capacity:
            raise OverflowError('Workspace capacity reached')
        self.stack.append(item)
    def pop(self):
        return self.stack.pop() if self.stack else None
    def depth(self):
        return len(self.stack)
    # alpha interference proxy: returns float 0-1
    def alpha_load(self):
        return self.depth()/self.capacity
