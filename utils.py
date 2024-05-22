import numpy as np
import torch

def numpy_to_tensor(array: np.ndarray) -> torch.tensor:
    # Convert to tensor
    tensor = torch.from_numpy(array).float()

    return tensor

def tensor_to_numpy(tensor: torch.tensor) -> np.ndarray:
    # Ensure tensor is on CPU
    if tensor.device != torch.device('cpu'):
        tensor = tensor.cpu()
    # Detach tensor if it requires gradients
    if tensor.requires_grad:
        tensor = tensor.detach()
    # Convert to NumPy
    return tensor.numpy()