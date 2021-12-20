import numpy as np

class Episodic_Memory:
    def __init__(self, n_workers, mem_size, mem_dim) -> None:
        self.mem_dim = mem_dim 
        self.mem_size = mem_size 
        self.n_workers = n_workers
        self.reset()
    
    def reset(self):
        self.pointer = 0
        self.memory = np.zeros((self.n_workers, self.mem_size, self.mem_dim))

    def push(self, memories):
        self.memory[:, self.pointer, :] = memories
        self.pointer += 1
    
    def generate_mask(self):
        mask = np.zeros((self.n_workers, self.mem_size))
        mask[:, :self.pointer] = 1
        return mask 
