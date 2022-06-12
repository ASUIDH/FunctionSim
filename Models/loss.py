import torch
def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)