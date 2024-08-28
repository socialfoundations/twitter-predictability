import torch

def allocated_memory():
    return torch.cuda.memory_reserved() / 1024 ** 3